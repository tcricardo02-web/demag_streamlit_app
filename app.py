import streamlit as st
import logging
from dataclasses import dataclass
from typing import List
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pint
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, Float, ForeignKey, DateTime
)
from sqlalchemy.orm import sessionmaker, relationship, declarative_base

# ----------------------------------------------------------
# Logger & Pint
# ----------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

# ----------------------------------------------------------
# Database setup
# ----------------------------------------------------------
Base = declarative_base()
engine = create_engine(
    "sqlite:///compressor.db",
    connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(bind=engine)

class PerformanceRun(Base):
    __tablename__ = "performance_run"
    id              = Column(Integer, primary_key=True)
    timestamp       = Column(DateTime, default=datetime.utcnow)
    mass_flow       = Column(Float)
    inlet_pressure  = Column(Float)
    inlet_temp      = Column(Float)
    outlet_pressure = Column(Float)
    total_kW        = Column(Float)
    total_BHP       = Column(Float)
    n_stages        = Column(Integer)
    details         = relationship(
        "StageDetailModel", back_populates="run", cascade="all, delete-orphan"
    )

class StageDetailModel(Base):
    __tablename__ = "stage_detail"
    id                    = Column(Integer, primary_key=True)
    run_id                = Column(Integer, ForeignKey("performance_run.id"))
    stage                 = Column(Integer)
    P_in_bar              = Column(Float)
    T_in_C                = Column(Float)
    P_out_bar             = Column(Float)
    T_out_C               = Column(Float)
    isentropic_efficiency = Column(Float)
    shaft_power_kW        = Column(Float)
    run                   = relationship("PerformanceRun", back_populates="details")

def init_db():
    Base.metadata.create_all(engine)
    logger.info("Database initialized")

# ----------------------------------------------------------
# Domain dataclasses & helpers
# ----------------------------------------------------------
@dataclass
class Throw:
    throw_number: int
    stage_assignment: int
    VVCP_pct: float
    clearance_pct: float

@dataclass
class Actuator:
    power_kW: float
    derate_percent: float
    air_cooler_fraction: float

@dataclass
class MotorCurve:
    rpm_points: List[float]
    power_kW:   List[float]

    def available_power(self, rpm: float) -> float:
        """ NumPy‐only linear interp + extrapolation """
        if rpm <= min(self.rpm_points):
            return self.power_kW[0]
        if rpm >= max(self.rpm_points):
            return self.power_kW[-1]
        idx = np.searchsorted(self.rpm_points, rpm)
        x0, x1 = self.rpm_points[idx-1], self.rpm_points[idx]
        y0, y1 = self.power_kW[idx-1], self.power_kW[idx]
        return y0 + (rpm - x0) * (y1 - y0) / (x1 - x0)

# ----------------------------------------------------------
# Calculation functions
# ----------------------------------------------------------
def clamp(x, a, b):
    return max(a, min(b, x))

def perform_performance_calculation(
    mass_flow,
    P_in,
    T_in,
    P_out,
    throws: List[Throw],
    actuator: Actuator,
    motor_curve: MotorCurve,
    n_stages: int
):
    # 1) Motor available (derated)
    avail_kw = motor_curve.available_power(motor_curve.current_rpm)
    avail_kw *= (1 - actuator.derate_percent / 100)

    # 2) Stagewise compression + cooling
    P = P_in.to(ureg.Pa).magnitude
    T = T_in.to(ureg.K).magnitude
    gamma, cp = 1.30, 2.0
    PR_total = P_out.to(ureg.Pa).magnitude / P
    PR_base  = PR_total**(1/n_stages)

    total_kW = 0.0
    details  = []

    for stage in range(1, n_stages+1):
        Pin_s  = P * (PR_base**(stage-1))
        Pout_s = Pin_s * PR_base

        assigned = [t for t in throws if t.stage_assignment==stage]
        vvcp = np.mean([t.VVCP_pct for t in assigned]) if assigned else 0.0
        clr  = np.mean([t.clearance_pct for t in assigned]) if assigned else 0.0

        eta = clamp(
            0.65 + 0.15*(vvcp/100) - 0.05*(vvcp/100) + 0.10*(clr/100),
            0.65, 0.92
        )

        T_is = T*(PR_base**((gamma-1)/gamma))
        Tout = T + (T_is - T)/eta
        dT   = Tout - T

        Wk = mass_flow * cp * dT / 1000
        total_kW += Wk

        details.append({
            "stage":                 stage,
            "P_in_bar":              Pin_s/1e5,
            "T_in_C":                T-273.15,
            "P_out_bar":             Pout_s/1e5,
            "T_out_C":               Tout-273.15,
            "isentropic_efficiency": eta,
            "shaft_power_kW":        Wk
        })

        # interstage cooler: 1% ΔP loss, reset T to 120 °F
        P    = Pout_s * 0.99
        Tout = (120-32)*5/9 + 273.15
        T    = Tout

    return {
        "total_kW":  total_kW,
        "total_BHP": total_kW * 1.34102,
        "details":   details
    }

# ----------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------
def main():
    st.set_page_config(page_title="Ariel7 Compressor", layout="wide")
    init_db()

    # Motor‐curve state
    if "motor_curve_pts" not in st.session_state:
        st.session_state.motor_curve_pts = {"rpm":[900,1200], "kW":[200,300]}

    tabs = st.tabs(["Processo","Equipamento","Report","Multi-Run"])

    # — Equipamento —
    with tabs[1]:
        st.header("Configuração do Compressor")
        n_stages = st.number_input("Número de Estágios", value=3, min_value=1, step=1)
        rpm      = st.number_input("Frame RPM",          value=900, min_value=100, step=10)
        stroke   = st.number_input("Stroke (m)",         value=0.12, format="%.3f")
        n_throws = st.number_input("Número de Throws",    value=3, min_value=1, step=1)

        throws: List[Throw] = []
        for i in range(1, n_throws+1):
            st.markdown(f"🔩 Throw {i}")
            sa   = st.selectbox(f"Estágio p/Throw {i}", list(range(1,n_stages+1)), key=f"stage_{i}")
            vvcp = st.slider(f"VVCP % #{i}", 0.0,100.0,90.0, key=f"vvcp_{i}")
            clr  = st.slider(f"Clearance % #{i}",0.0,100.0,2.0,  key=f"clr_{i}")
            throws.append(Throw(i,sa,vvcp,clr))

        actuator = Actuator(
            st.number_input("Potência Atuador (kW)", value=250.0),
            st.number_input("Derate (%)",             value=5.0),
            st.number_input("Air-Cooler (%)",         value=25.0)
        )

        st.markdown("---")
        st.subheader("Motor & Curva de Potência")
        _ = st.radio("Tipo de Motor", ["Elétrico","Gás Natural"])

        dfc   = pd.DataFrame(st.session_state.motor_curve_pts)
        ed    = st.data_editor(dfc, num_rows="dynamic")
        st.session_state.motor_curve_pts = {"rpm":ed["rpm"].tolist(), "kW":ed["kW"].tolist()}

        mc = MotorCurve(
            rpm_points=st.session_state.motor_curve_pts["rpm"],
            power_kW=  st.session_state.motor_curve_pts["kW"]
        )
        mc.current_rpm = rpm

        st.session_state.eq_config = {
            "n_stages":    n_stages,
            "throws":      throws,
            "actuator":    actuator,
            "motor_curve": mc
        }
        st.success("Configuração salva.")

    # — Processo —
    with tabs[0]:
        st.header("Processo & Diagrama P–T")
        c1,c2 = st.columns(2)
        pin  = c1.number_input("P sucção (psig)", 30.0)
        tin  = c1.number_input("T sucção (°F)",  77.0)
        pout = c2.number_input("P descarga (psig)",60.0)
        mf   = c2.number_input("Fluxo (kg/s)",      12.0)

        st.session_state.process = {
            "P_in":  Q_(pin*6894.76, ureg.Pa),
            "T_in":  Q_((tin-32)*5/9+273.15, ureg.K),
            "P_out": Q_(pout*6894.76, ureg.Pa),
            "mf":     mf
        }

        if "eq_config" in st.session_state:
            cfg = st.session_state.eq_config
            pr  = st.session_state.process
            out = perform_performance_calculation(
                pr["mf"], pr["P_in"], pr["T_in"], pr["P_out"],
                cfg["throws"], cfg["actuator"], cfg["motor_curve"], cfg["n_stages"]
            )

            fig = go.Figure()
            for d in out["details"]:
                fig.add_trace(go.Scatter(
                    x=[d["P_in_bar"], d["P_out_bar"]],
                    y=[d["T_in_C"], d["T_out_C"]],
                    mode="lines+markers",
                    name=f"Stage {d['stage']}"
                ))
                # cooler dash
                fig.add_trace(go.Scatter(
                    x=[d["P_out_bar"], d["P_out_bar"]*0.99],
                    y=[d["T_out_C"], 120.0],
                    mode="lines",
                    line=dict(dash="dash"), showlegend=False
                ))

            fig.update_layout(xaxis_title="Pressure (bar)", yaxis_title="Temperature (°C)")
            st.plotly_chart(fig, use_container_width=True)

    # — Multi-Run —
    with tabs[3]:
        st.header("Multi-Run Sweep")
        cfg = st.session_state.get("eq_config", {})
        pr  = st.session_state.get("process", {})

        cA,cB = st.columns(2)
        omin = cA.number_input("P_out min (psig)", 40.0)
        omax = cA.number_input("P_out max (psig)",100.0)
        dP   = cA.number_input("ΔP step",           5.0)
        rmin = cB.number_input("RPM min",         600, step=10)
        rmax = cB.number_input("RPM max",        1200, step=10)
        dr   = cB.number_input("ΔRPM",            100, step=10)

        if st.button("Executar Multi-Run"):
            rows = []
            for P in np.arange(omin, omax+dP/2, dP):
                pout_loop = Q_(P*6894.76, ureg.Pa)
                for R in np.arange(rmin, rmax+dr/2, dr):
                    cfg["motor_curve"].current_rpm = R
                    out = perform_performance_calculation(
                        pr["mf"], pr["P_in"], pr["T_in"], pout_loop,
                        cfg["throws"], cfg["actuator"], cfg["motor_curve"], cfg["n_stages"]
                    )
                    rows.append({"P_out_psig": P, "RPM": R, "Flow (kg/s)": pr["mf"], "BHP": out["total_BHP"]})

            dfm = pd.DataFrame(rows)
            st.plotly_chart(px.line(dfm, x="P_out_psig", y="Flow (kg/s)", color="RPM", markers=True), use_container_width=True)
            st.plotly_chart(px.line(dfm, x="P_out_psig", y="BHP",           color="RPM", markers=True), use_container_width=True)

    # — Report —
    with tabs[2]:
        st.header("Report")
        st.markdown("Genere no Processo/Multi-Run e então salve em CSV/DB aqui.")

if __name__ == "__main__":
    main()
