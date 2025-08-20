import streamlit as st
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional
import plotly.graph_objects as go
import pint
from sqlalchemy import create_engine, Column, Integer, Float, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, declarative_base

# ------------------------------------------------------------------------------
# Logger e Pint (unidades)
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

# ------------------------------------------------------------------------------
# Banco de dados (SQLAlchemy)
# ------------------------------------------------------------------------------
DB_PATH = "sqlite:///compressor.db"
Base = declarative_base()
engine = create_engine(DB_PATH, echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

class FrameModel(Base):
    __tablename__ = "frame"
    id = Column(Integer, primary_key=True, index=True)
    rpm = Column(Float)
    stroke_m = Column(Float)
    n_throws = Column(Integer)
    throws = relationship("ThrowModel", back_populates="frame")

class ThrowModel(Base):
    __tablename__ = "throw"
    id = Column(Integer, primary_key=True, index=True)
    frame_id = Column(Integer, ForeignKey("frame.id"))
    throw_number = Column(Integer)
    bore_m = Column(Float)
    clearance_m = Column(Float)
    VVCP = Column(Float)
    SACE = Column(Float)
    SAHE = Column(Float)
    frame = relationship("FrameModel", back_populates="throws")

class ActuatorModel(Base):
    __tablename__ = "actuator"
    id = Column(Integer, primary_key=True, index=True)
    power_available_kW = Column(Float)
    derate_percent = Column(Float)
    air_cooler_fraction = Column(Float)

def init_db():
    Base.metadata.create_all(bind=engine)
    logger.info("Banco de dados inicializado.")

# ------------------------------------------------------------------------------
# Dataclasses (domínio)
# ------------------------------------------------------------------------------
@dataclass
class Frame:
    rpm: float
    stroke: float       # m
    n_throws: int

@dataclass
class Throw:
    throw_number: int
    bore: float         # m
    clearance: float    # m
    VVCP: float         # %
    SACE: float         # %
    SAHE: float         # %
    throw_id: Optional[int] = None

@dataclass
class Actuator:
    power_kW: float
    derate_percent: float
    air_cooler_fraction: float

@dataclass
class Motor:
    power_kW: float     # kW

# ------------------------------------------------------------------------------
# Funções de cálculo
# ------------------------------------------------------------------------------
def clamp(n, a, b):
    return max(a, min(n, b))

def perform_performance_calculation(
    mass_flow: float,
    inlet_pressure: Q_,
    inlet_temperature: Q_,
    n_stages: int,
    PR_total: float,
    throws: List[Throw],
    stage_mapping: Dict[int, List[int]],
    actuator: Actuator,
) -> Dict:
    m_dot = mass_flow
    P_in = inlet_pressure.to(ureg.Pa).magnitude
    T_in = inlet_temperature.to(ureg.K).magnitude
    n = max(n_stages, 1)
    PR_base = PR_total ** (1.0 / n)
    gamma = 1.30
    cp = 2.0  # kJ/(kg·K)

    throws_by_number = {t.throw_number: t for t in throws}
    total_W_kW = 0.0
    stage_details = []

    for stage in range(1, n+1):
        P_in_stage = P_in * (PR_base ** (stage-1))
        P_out_stage = P_in_stage * PR_base
        assigned = stage_mapping.get(stage, [])
        if assigned:
            SACE_avg = sum(throws_by_number[t].SACE for t in assigned if t in throws_by_number) / len(assigned)
            VVCP_avg = sum(throws_by_number[t].VVCP for t in assigned if t in throws_by_number) / len(assigned)
            SAHE_avg = sum(throws_by_number[t].SAHE for t in assigned if t in throws_by_number) / len(assigned)
        else:
            SACE_avg = VVCP_avg = SAHE_avg = 0.0

        eta_isent = 0.65 + 0.15*(SACE_avg/100.0) - 0.05*(VVCP_avg/100.0) + 0.10*(SAHE_avg/100.0)
        eta_isent = clamp(eta_isent, 0.65, 0.92)

        T_out_isent = T_in * (PR_base ** ((gamma-1.0)/gamma))
        T_out_actual = T_in + (T_out_isent - T_in) / max(eta_isent, 1e-6)
        delta_T = T_out_actual - T_in

        W_stage = m_dot * cp * delta_T / 1000.0
        total_W_kW += W_stage

        stage_details.append({
            "stage": stage,
            "P_in_bar": P_in_stage/1e5,
            "P_out_bar": P_out_stage/1e5,
            "PR": PR_base,
            "T_in_C": T_in-273.15,
            "T_out_C": T_out_actual-273.15,
            "isentropic_efficiency": eta_isent,
            "shaft_power_kW": W_stage,
            "shaft_power_BHP": W_stage * 1.34102
        })

        T_in = T_out_actual

    outputs = {
        "mass_flow_kg_s": m_dot,
        "inlet_pressure_bar": P_in/1e5,
        "inlet_temperature_C": inlet_temperature.to(ureg.degC).magnitude,
        "n_stages": n_stages,
        "total_shaft_power_kW": total_W_kW,
        "total_shaft_power_BHP": total_W_kW * 1.34102,
        "stage_details": stage_details,
        "a_Ariel7_compatible": {
            "stages": stage_details,
            "total_shaft_power_kW": total_W_kW,
            "total_shaft_power_BHP": total_W_kW * 1.34102
        }
    }
    return outputs

# ------------------------------------------------------------------------------
# Diagrama interativo (Plotly)
# ------------------------------------------------------------------------------
def generate_diagram(frame: Frame, throws: List[Throw], actuator: Actuator, motor: Motor) -> go.Figure:
    """
    Monta um diagrama representando:
      - Motor (à esquerda), com potência em BHP;
      - Frame do compressor (centro);
      - Throws abaixo do frame;
      - Atuador à direita.
    """
    fig = go.Figure()
    canvas_w, canvas_h = 900, 350

    # Motor (esquerda)
    mx, my, mw, mh = 30, canvas_h/2 - 25, 100, 50
    fig.add_shape(
        type="rect",
        x0=mx, y0=my, x1=mx+mw, y1=my+mh,
        line=dict(color="MediumPurple"),
        fillcolor="Lavender"
    )
    fig.add_annotation(
        x=mx+mw/2, y=my+mh/2,
        text=f"Motor<br>{motor.power_kW * 1.34102:.0f} BHP",
        showarrow=False, align="center"
    )

    # Frame (centro)
    fx, fy, fw, fh = mx + mw + 50, canvas_h/2 - 25, 200, 50
    fig.add_shape(
        type="rect",
        x0=fx, y0=fy, x1=fx+fw, y1=fy+fh,
        line=dict(color="RoyalBlue"),
        fillcolor="LightSkyBlue"
    )
    fig.add_annotation(
        x=fx+fw/2, y=fy+fh/2,
        text=f"Frame<br>RPM: {frame.rpm:.0f}",
        showarrow=False, align="center"
    )

    # Throws (abaixo do frame)
    n = len(throws)
    spacing = fw / n if n > 0 else 0
    for t in throws:
        idx = t.throw_number - 1
        tx = fx + idx * spacing + spacing/4
        ty = fy + fh + 20
        tw, th = spacing/2, 30
        fig.add_shape(
            type="rect",
            x0=tx, y0=ty, x1=tx+tw, y1=ty+th,
            line=dict(color="DarkOrange"),
            fillcolor="Moccasin"
        )
        fig.add_annotation(
            x=tx+tw/2, y=ty+th/2,
            text=f"Throw {t.throw_number}",
            showarrow=False, font=dict(size=10)
        )

    # Atuador (direita)
    ax, ay, aw, ah = fx + fw + 50, canvas_h/2 - 20, 120, 60
    fig.add_shape(
        type="rect",
        x0=ax, y0=ay, x1=ax+aw, y1=ay+ah,
        line=dict(color="SaddleBrown"),
        fillcolor="PeachPuff"
    )
    fig.add_annotation(
        x=ax+aw/2, y=ay+ah/2,
        text=f"Acionador<br>{actuator.power_kW:.0f} kW",
        showarrow=False, align="center"
    )

    fig.update_layout(
        width=canvas_w,
        height=canvas_h,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


# ------------------------------------------------------------------------------
# UI com Streamlit
# ------------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Compressor Ariel7-style", layout="wide")
    st.title("Calculadora de Performance de Compressor (Estilo Ariel 7)")

    init_db()
    if "unit_system" not in st.session_state:
        st.session_state["unit_system"] = "SI"

    with st.sidebar:
        st.header("Geral")
        unit = st.selectbox("Sistema de unidades", ["SI", "Metric"], index=0)
        st.session_state["unit_system"] = unit
        if st.button("Resetar DB"):
            import os
            if os.path.exists("compressor.db"):
                os.remove("compressor.db")
            init_db()
            st.success("Banco de dados reinicializado.")

    tabs = st.tabs(["Processo", "Configuração do Equipamento"])

    # Aba Processo
    with tabs[0]:
        st.header("Processo")
        if st.session_state["unit_system"] == "SI":
            p = st.number_input("Pressão incial (Pa)",  value=200000.0)
            t = st.number_input("Temperatura (K)",    value=298.15)
        else:
            p = st.number_input("Pressão (bar)",      value=2.0) * 1e5
            t = st.number_input("Temp (°C)",          value=25.0) + 273.15

        mf = st.number_input("Fluxo mássico (kg/s)", value=12.0)
        ns = st.number_input("Nº de estágios",     min_value=1, value=3)
        pr = st.number_input("PR total",           min_value=1.0, value=2.5)

        if st.button("Calcular (Processo)"):
            out = perform_performance_calculation(
                mass_flow=mf,
                inlet_pressure=Q_(p, ureg.Pa),
                inlet_temperature=Q_(t, ureg.K),
                n_stages=ns,
                PR_total=pr,
                throws=[],
                stage_mapping={},
                actuator=Actuator(0,0,0)
            )
            st.json(out)

    # Aba Configuração
    with tabs[1]:
        st.header("Configuração do Equipamento")

        # Frame
        rpm = st.number_input("RPM do Frame", min_value=100, max_value=3000, value=900, step=10)
        if st.session_state["unit_system"] == "SI":
            stroke = st.number_input("Stroke (m)", value=0.12, step=0.01)
        else:
            stroke = st.number_input("Stroke (mm)", value=120, step=1) / 1000.0
        n_throws = st.number_input("Total de Throws", min_value=1, max_value=20, value=3)

        frame = Frame(rpm=rpm, stroke=stroke, n_throws=int(n_throws))

        # Throws
        throws: List[Throw] = []
        for i in range(1, int(n_throws)+1):
            st.markdown(f"**Throw {i}**")
            if st.session_state["unit_system"] == "SI":
                bore = st.number_input(f"Bore (m) #{i}", value=0.08, step=0.001, key=f"b{i}")
                clr = st.number_input(f"Clearance (m) #{i}", value=0.002, step=0.0005, key=f"c{i}")
            else:
                bore = st.number_input(f"Bore (mm) #{i}", value=80, key=f"b{i}") / 1000.0
                clr = st.number_input(f"Clearance (mm) #{i}", value=2, key=f"c{i}") / 1000.0
            vvcp = st.number_input(f"VVCP (%) #{i}", value=90, key=f"v{i}")
            sace = st.number_input(f"SACE (%) #{i}", value=80, key=f"s{i}")
            sahe = st.number_input(f"SAHE (%) #{i}", value=60, key=f"h{i}")
            throws.append(Throw(i, bore, clr, vvcp, sace, sahe))

        # Mapeamento multi-throw por estágio
        ns = st.number_input("Nº de estágios (para mapear)", min_value=1, max_value=12, value=3)
        stage_map: Dict[int, List[int]] = {}
        for s in range(1, ns+1):
            opts = [f"Throw {t.throw_number}" for t in throws]
            sel = st.multiselect(f"Estágio {s} recebe:", opts, key=f"m{s}")
            ids = [int(x.split()[1]) for x in sel]
            stage_map[s] = ids

        # Atuador e Motor
        pw = st.number_input("Potência Atuador (kW)", value=250.0)
        dr = st.number_input("Derate (%)", value=5.0)
        acf = st.number_input("Air Cooler (%)", value=25.0)
        actuator = Actuator(pw, dr, acf)

        mk = st.number_input("Potência Motor (kW)", value=300.0)
        motor = Motor(mk)

        # Diagrama
        st.markdown("---")
        st.subheader("Diagrama do Equipamento")
        fig = generate_diagram(frame, throws, actuator, motor)
        st.plotly_chart(fig, use_container_width=True)

        # Salvar e calcular
        if st.button("Salvar Configuração e Calcular"):
            db = SessionLocal()
            fm = FrameModel(rpm=frame.rpm, stroke_m=frame.stroke, n_throws=frame.n_throws)
            db.add(fm); db.commit(); db.refresh(fm)
            for t in throws:
                tm = ThrowModel(
                    frame_id=fm.id, throw_number=t.throw_number,
                    bore_m=t.bore, clearance_m=t.clearance,
                    VVCP=t.VVCP, SACE=t.SACE, SAHE=t.SAHE
                )
                db.add(tm)
            am = ActuatorModel(power_available_kW=pw, derate_percent=dr, air_cooler_fraction=acf)
            db.add(am)
            db.commit(); db.close()

            out = perform_performance_calculation(
                mass_flow=12.0,
                inlet_pressure=Q_(6000000, ureg.Pa),
                inlet_temperature=Q_(298.15, ureg.K),
                n_stages=ns,
                PR_total=2.5,
                throws=throws,
                stage_mapping=stage_map,
                actuator=actuator
            )
            out["frame_rpm"] = rpm
            st.success("Configuração salva e outputs calculados")
            st.json(out)

if __name__ == "__main__":
    main()
