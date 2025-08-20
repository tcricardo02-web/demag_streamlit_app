import streamlit as st
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional
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
engine = create_engine('sqlite:///compressor.db', connect_args={'check_same_thread': False})
SessionLocal = sessionmaker(bind=engine)

# Models
class FrameModel(Base):
    __tablename__ = 'frame'
    id = Column(Integer, primary_key=True)
    rpm = Column(Float)
    stroke_m = Column(Float)
    n_throws = Column(Integer)
    throws = relationship('ThrowModel', back_populates='frame')

class ThrowModel(Base):
    __tablename__ = 'throw'
    id = Column(Integer, primary_key=True)
    frame_id = Column(Integer, ForeignKey('frame.id'))
    throw_number = Column(Integer)
    stage_assignment = Column(Integer)
    bore_m = Column(Float)
    clearance_m = Column(Float)
    VVCP = Column(Float)
    SACE = Column(Float)
    SAHE = Column(Float)
    frame = relationship('FrameModel', back_populates='throws')

class ActuatorModel(Base):
    __tablename__ = 'actuator'
    id = Column(Integer, primary_key=True)
    power_available_kW = Column(Float)
    derate_percent = Column(Float)
    air_cooler_fraction = Column(Float)

class PerformanceRun(Base):
    __tablename__ = 'performance_run'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    mass_flow = Column(Float)
    inlet_pressure = Column(Float)
    inlet_temp = Column(Float)
    outlet_pressure = Column(Float)
    total_kW = Column(Float)
    total_BHP = Column(Float)
    n_stages = Column(Integer)
    details = relationship('StageDetailModel', back_populates='run')

class StageDetailModel(Base):
    __tablename__ = 'stage_detail'
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey('performance_run.id'))
    stage = Column(Integer)
    P_in_bar = Column(Float)
    P_out_bar = Column(Float)
    isentropic_efficiency = Column(Float)
    shaft_power_kW = Column(Float)
    shaft_power_BHP = Column(Float)
    run = relationship('PerformanceRun', back_populates='details')

def init_db():
    Base.metadata.create_all(engine)
    logger.info('Database initialized')

# ----------------------------------------------------------
# Domain dataclasses
# ----------------------------------------------------------
@dataclass
class Frame:
    rpm: float
    stroke: float
    n_throws: int

@dataclass
class Throw:
    throw_number: int
    stage_assignment: int
    bore: float
    clearance: float
    VVCP: float
    SACE: float
    SAHE: float

@dataclass
class Actuator:
    power_kW: float
    derate_percent: float
    air_cooler_fraction: float

@dataclass
class Motor:
    power_kW: float

# ----------------------------------------------------------
# Calculation functions
# ----------------------------------------------------------
def clamp(x,a,b): return max(a, min(b, x))

def estimate_n_stages(PR_total: float, max_pr_stage: float=4.16) -> int:
    return int(np.ceil(np.log(PR_total) / np.log(max_pr_stage)))

def perform_performance_calculation(
    mass_flow, P_in, T_in, P_out, throws: List[Throw], actuator: Actuator
):
    m_dot = mass_flow
    P1 = P_in.to(ureg.Pa).magnitude
    P2 = P_out.to(ureg.Pa).magnitude
    PR_total = P2 / P1
    n_stages = estimate_n_stages(PR_total)
    PR_base = PR_total ** (1/n_stages)

    gamma, cp = 1.30, 2.0
    total_W_kW = 0.0
    details = []
    T = T_in.to(ureg.K).magnitude

    for stage in range(1, n_stages+1):
        Pin_s = P1 * (PR_base**(stage-1))
        Pout_s = Pin_s * PR_base
        assigned = [t for t in throws if t.stage_assignment == stage]
        if assigned:
            SACE = np.mean([t.SACE for t in assigned])
            VVCP = np.mean([t.VVCP for t in assigned])
            SAHE = np.mean([t.SAHE for t in assigned])
        else:
            SACE = VVCP = SAHE = 0.0

        eta = clamp(0.65 + 0.15*(SACE/100) - 0.05*(VVCP/100) + 0.10*(SAHE/100), 0.65, 0.92)
        Tis = T * (PR_base**((gamma-1)/gamma))
        Tout = T + (Tis - T)/eta
        dT = Tout - T
        Wk = m_dot * cp * dT / 1000
        total_W_kW += Wk

        details.append({
            'stage': stage,
            'P_in_bar': Pin_s/1e5,
            'P_out_bar': Pout_s/1e5,
            'isentropic_efficiency': eta,
            'shaft_power_kW': Wk,
            'shaft_power_BHP': Wk * 1.34102
        })
        T = Tout

    total_BHP = total_W_kW * 1.34102
    return {
        'mass_flow': m_dot,
        'inlet_bar': P1/1e5,
        'outlet_bar': P2/1e5,
        'PR_total': PR_total,
        'n_stages': n_stages,
        'total_kW': total_W_kW,
        'total_BHP': total_BHP,
        'details': details
    }

# ----------------------------------------------------------
# Diagram function
# ----------------------------------------------------------
def generate_equipment_diagram(frame: Frame, throws: List[Throw], actuator: Actuator, motor: Motor):
    fig = go.Figure()
    W, H = 900, 400

    # Motor
    mx, my, mw, mh = 20, H/2-30, 120, 60
    fig.add_shape(type='rect', x0=mx, y0=my, x1=mx+mw, y1=my+mh,
                  line=dict(color='DarkSlateBlue'), fillcolor='MediumPurple')
    fig.add_annotation(x=mx+mw/2, y=my+mh/2, showarrow=False, align='center',
                       text=f"Motor\n{motor.power_kW*1.34102:.0f} BHP")

    # Frame
    fx, fy, fw, fh = mx+mw+60, H/2-25, 240, 50
    fig.add_shape(type='rect', x0=fx, y0=fy, x1=fx+fw, y1=fy+fh,
                  line=dict(color='RoyalBlue'), fillcolor='LightSkyBlue')
    fig.add_annotation(x=fx+fw/2, y=fy+fh/2, showarrow=False, align='center',
                       text=f"Frame\nRPM {frame.rpm:.0f}")

    # Throws
    spacing = fw / max(frame.n_throws, 1)
    for t in throws:
        idx = t.throw_number - 1
        tx = fx + idx*spacing + spacing*0.1
        ty = fy + fh + 20
        fig.add_shape(type='rect', x0=tx, y0=ty, x1=tx+spacing*0.8, y1=ty+40,
                      line=dict(color='DarkOrange'), fillcolor='Moccasin')
        fig.add_annotation(x=tx+spacing*0.4, y=ty+20, showarrow=False, font=dict(size=10),
                           text=f"Throw {t.throw_number}\nStg {t.stage_assignment}")

    # Actuator
    ax, ay, aw, ah = fx+fw+60, H/2-30, 120, 60
    fig.add_shape(type='rect', x0=ax, y0=ay, x1=ax+aw, y1=ay+ah,
                  line=dict(color='SaddleBrown'), fillcolor='PeachPuff')
    fig.add_annotation(x=ax+aw/2, y=ay+ah/2, showarrow=False, align='center',
                       text=f"Acionador\n{actuator.power_kW:.0f} kW")

    fig.update_layout(width=W, height=H,
                      xaxis=dict(visible=False),
                      yaxis=dict(visible=False),
                      margin=dict(l=10, r=10, t=10, b=10))
    return fig

# ----------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------
def main():
    st.set_page_config(page_title='Compressor Ariel7', layout='wide')
    init_db()

    if 'unit' not in st.session_state:
        st.session_state['unit'] = 'SI'
    st.sidebar.selectbox('Unidades', ['SI', 'Metric'], key='unit')

    tabs = st.tabs(['Processo', 'Equipamento', 'Report', 'Multi-Run'])

    # Processo tab
    with tabs[0]:
        st.header('Processo')
        c1, c2 = st.columns(2)
        if st.session_state.unit == 'SI':
            pin = Q_(c1.number_input('P sucção (Pa)', 2e5), ureg.Pa)
            tin = Q_(c1.number_input('T sucção (K)', 298.15), ureg.K)
        else:
            pin = Q_(c1.number_input('P sucção (bar)', 2.0)*1e5, ureg.Pa)
            tin = Q_(c1.number_input('T sucção (°C)', 25.0)+273.15, ureg.K)

        pout = Q_(c2.number_input('P descarga (bar)', 4.0)*1e5, ureg.Pa)
        mf = c2.number_input('Fluxo (kg/s)', 12.0)
        PRtot = (pout/pin).magnitude
        nest = estimate_n_stages(PRtot)
        st.markdown(f'PR_total: {PRtot:.2f} (estágios: {nest})')
        est = perform_performance_calculation(mf, pin, tin, pout, [], Actuator(0,0,0))
        st.markdown(f"Estimativa kW: {est['total_kW']:.1f}, BHP: {est['total_BHP']:.1f}")

    # Equipamento tab
    with tabs[1]:
        st.header('Configuração do Equipamento')
        rpm = st.number_input('RPM Frame', 900)
        stroke = st.number_input(
            'Stroke (mm)' if st.session_state.unit=='Metric' else 'Stroke (m)',
            120 if st.session_state.unit=='Metric' else 0.12
        )
        stroke_m = stroke * (0.001 if st.session_state.unit=='Metric' else 1)
        n_thr = st.number_input('Número de Throws', 3, min_value=1)
        frame = Frame(rpm, stroke_m, n_thr)

        throws: List[Throw] = []
        for i in range(1, n_thr+1):
            st.markdown(f'**Throw {i}**')
            bore = st.number_input(f'Bore ({"mm" if st.session_state.unit=="Metric" else "m"})', 
                                   80 if st.session_state.unit=='Metric' else 0.08, key=f'b{i}')
            bore_m = bore * (0.001 if st.session_state.unit=='Metric' else 1)
            clr = st.number_input(f'Clearance ({"mm" if st.session_state.unit=="Metric" else "m"})', 
                                  2 if st.session_state.unit=='Metric' else 0.002, key=f'c{i}')
            clr_m = clr * (0.001 if st.session_state.unit=='Metric' else 1)
            stg = st.number_input(f'Estágio p/ Throw {i}', 1, min_value=1, key=f'st{i}')
            vvcp = st.number_input(f'VVCP #{i}', 90, key=f'v{i}')
            sace = st.number_input(f'SACE #{i}', 80, key=f's{i}')
            sahe = st.number_input(f'SAHE #{i}', 60, key=f'h{i}')
            throws.append(Throw(i, stg, bore_m, clr_m, vvcp, sace, sahe))

        pw = st.number_input('Potência Atuador (kW)', 250.0)
        dr = st.number_input('Derate (%)', 5.0)
        ac = st.number_input('Air Cooler (%)', 25.0)
        actuator = Actuator(pw, dr, ac)

        mk = st.number_input('Potência Motor (kW)', 300.0)
        motor = Motor(mk)

        st.subheader('Diagrama')
        fig = generate_equipment_diagram(frame, throws, actuator, motor)
        st.plotly_chart(fig, use_container_width=True)

    # Report tab
    with tabs[2]:
        st.header('Performance Report')
        if st.button('Salvar e Gerar Report'):
            out = perform_performance_calculation(mf, pin, tin, pout, throws, actuator)
            db = SessionLocal()
            run = PerformanceRun(
                mass_flow=mf,
                inlet_pressure=pin.to(ureg.Pa).magnitude,
                inlet_temp=tin.to(ureg.K).magnitude,
                outlet_pressure=pout.to(ureg.Pa).magnitude,
                total_kW=out['total_kW'],
                total_BHP=out['total_BHP'],
                n_stages=out['n_stages']
            )
            db.add(run); db.commit(); db.refresh(run)
            for d in out['details']:
                sd = StageDetailModel(
                    run_id=run.id,
                    stage=d['stage'],
                    P_in_bar=d['P_in_bar'],
                    P_out_bar=d['P_out_bar'],
                    isentropic_efficiency=d['isentropic_efficiency'],
                    shaft_power_kW=d['shaft_power_kW'],
                    shaft_power_BHP=d['shaft_power_BHP']
                )
                db.add(sd)
            db.commit(); db.close()

            df = pd.DataFrame(out['details'])
            st.subheader('Estágios')
            st.dataframe(df)
            st.markdown(f"**Total kW:** {out['total_kW']:.1f}  **Total BHP:** {out['total_BHP']:.1f}")

    # Multi-Run tab
    with tabs[3]:
        st.header('Multi-Run')
        cA, cB = st.columns(2)
        pmin = cA.number_input('P_out min (bar)', 4.0)
        pmax = cA.number_input('P_out max (bar)', 10.0)
        dp   = cA.number_input('ΔP step (bar)', 1.0)
        rpm_min = cB.number_input('RPM min', 600)
        rpm_max = cB.number_input('RPM max', 1200)
        drpm    = cB.number_input('ΔRPM', 200)

        if st.button('Executar Multi-Run'):
            rows = []
            for P in np.arange(pmin, pmax+1e-6, dp):
                pout_loop = Q_(P*1e5, ureg.Pa)
                for r in np.arange(rpm_min, rpm_max+1, drpm):
                    frame.rpm = r
                    out = perform_performance_calculation(mf, pin, tin, pout_loop, throws, actuator)
                    rows.append({'P_out': P, 'RPM': r, 'flow': mf, 'BHP': out['total_BHP']})
            dfm = pd.DataFrame(rows)
            fig1 = px.line(dfm, x='P_out', y='flow', color='RPM', markers=True,
                           title='Fluxo vs P_out')
            fig2 = px.line(dfm, x='P_out', y='BHP', color='RPM', markers=True,
                           title='BHP vs P_out')
            st.plotly_chart(fig1, use_container_width=True)
            st.plotly_chart(fig2, use_container_width=True)

if __name__ == '__main__':
    main()
