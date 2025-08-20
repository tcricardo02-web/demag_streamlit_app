import os
import zipfile

# Define the project directory and zip file path
project_folder = '/home/user/demag_streamlit_app'
zip_file_path = '/home/user/demag_streamlit_app.zip'

# Create the project directory if it doesn't exist
os.makedirs(project_folder, exist_ok=True)

# -------------------------------------------------------------------
# Write app.py
# -------------------------------------------------------------------
app_code = r'''import streamlit as st
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import pint

from sqlalchemy import create_engine, Column, Integer, Float, String, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, declarative_base

# ------------------------------------------------------------------------------
# Configuração de logger e unidades (Pint)
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

# ------------------------------------------------------------------------------
# Configuração do banco de dados com SQLAlchemy
# ------------------------------------------------------------------------------
DB_PATH = "sqlite:///compressor.db"
Base = declarative_base()
engine = create_engine(DB_PATH, echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

# Modelos ORM simplificados
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
# Domínio: Dataclasses para entidades
# ------------------------------------------------------------------------------
@dataclass
class Frame:
    rpm: float
    stroke: float  # em metros (SI)
    n_throws: int

@dataclass
class Throw:
    throw_number: int
    bore: float       # metros
    clearance: float  # metros
    VVCP: float       # %
    SACE: float       # %
    SAHE: float       # %
    throw_id: Optional[int] = None

@dataclass
class Actuator:
    power_kW: float
    derate_percent: float
    air_cooler_fraction: float

@dataclass
class Motor:
    power_kW: float   # Potência do motor (para diagrama)

# ------------------------------------------------------------------------------
# Cálculos de performance (inspirado no Ariel 7)
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
    stage_mapping: Dict[int, List[int]],  # Agora, cada estágio pode ter mais de um throw
    actuator: Actuator,
) -> Dict:
    """
    Calcula outputs de performance para cada estágio.
    Entradas em SI e a combinação dos parâmetros dos throws atribuídos é feita via média.
    """
    m_dot = mass_flow  # kg/s
    P_in = inlet_pressure.to(ureg.Pa).magnitude
    T_in = inlet_temperature.to(ureg.K).magnitude

    n = max(n_stages, 1)
    PR_base = PR_total ** (1.0 / n)
    
    gamma = 1.30
    cp = 2.0  # kJ/(kg*K)
    
    stage_details = []
    total_W_kW = 0.0
    
    # Cria um dicionário para acesso rápido: chave = throw_number, valor = Throw
    throws_by_number = {t.throw_number: t for t in throws}
    
    for stage in range(1, n + 1):
        # Entrada e saída do estágio
        P_in_stage = P_in * (PR_base ** (stage - 1))
        P_out_stage = P_in_stage * PR_base
        
        # Obter os throws atribuídos (lista de IDs) e combinar parâmetros (média)
        assigned = stage_mapping.get(stage, [])
        if assigned:
            SACE_avg = sum(throws_by_number[t].SACE for t in assigned if t in throws_by_number) / len(assigned)
            VVCP_avg = sum(throws_by_number[t].VVCP for t in assigned if t in throws_by_number) / len(assigned)
            SAHE_avg = sum(throws_by_number[t].SAHE for t in assigned if t in throws_by_number) / len(assigned)
        else:
            SACE_avg = VVCP_avg = SAHE_avg = 0.0
        
        # Eficiência isentrópica influenciada pelos parâmetros (média)
        eta_isent = 0.65 + 0.15 * (SACE_avg / 100.0) - 0.05 * (VVCP_avg / 100.0) + 0.10 * (SAHE_avg / 100.0)
        eta_isent = clamp(eta_isent, 0.65, 0.92)
        
        # Cálculo isentrópico e real
        T_out_isent = T_in * (PR_base ** ((gamma - 1.0) / gamma))
        T_out_actual = T_in + (T_out_isent - T_in) / max(eta_isent, 1e-6)
        delta_T = T_out_actual - T_in
        
        W_stage = m_dot * cp * delta_T / 1000.0   # kW
        total_W_kW += W_stage
        
        stage_details.append({
            "stage": stage,
            "P_in_bar": P_in_stage / 1e5,
            "P_out_bar": P_out_stage / 1e5,
            "PR": PR_base,
            "T_in_C": T_in - 273.15,
            "T_out_C": T_out_actual - 273.15,
            "isentropic_efficiency": eta_isent,
            "shaft_power_kW": W_stage,
            "shaft_power_BHP": W_stage * 1.34102  # conversão: 1 kW ≈ 1.34102 BHP
        })
        
        # Para próximo estágio, a saída atual vira a entrada
        T_in = T_out_actual
    
    outputs = {
        "mass_flow_kg_s": m_dot,
        "inlet_pressure_bar": P_in / 1e5,
        "inlet_temperature_C": inlet_temperature.to(ureg.degC).magnitude,
        "n_stages": n_stages,
        "total_shaft_power_kW": total_W_kW,
        "total_shaft_power_BHP": total_W_kW * 1.34102,
        "stage_details": stage_details,
        "a_Ariel7_compatible": {
            "stages": stage_details,
            "total_shaft_power_kW": total_W_kW,
            "total_shaft_power_BHP": total_W_kW * 1.34102,
        }
    }
    return outputs

# ------------------------------------------------------------------------------
# Diagrama interativo com Plotly: Motor, Frame e Throws (estilo Ariel7)
# ------------------------------------------------------------------------------
def generate_diagram(frame: Frame, throws: List[Throw], actuator: Actuator, motor: Motor) -> go.Figure:
    """
    Monta um diagrama representando:
      - O motor (à esquerda) com potência em BHP;
      - O frame do compressor (no centro);
      - Os throws abaixo do frame (podem ser mais de um estágio, mas aqui exibimos todos os throws cadastrados).
    """
    fig = go.Figure()
    
    canvas_width = 900
    canvas_height = 350
    
    # Posição do motor (à esquerda)
    motor_x = 30
    motor_y = canvas_height / 2 - 25
    motor_width = 100
    motor_height = 50
    fig.add_shape(
        type="rect",
        x0=motor_x, y0=motor_y,
        x1=motor_x+motor_width, y1=motor_y+motor_height,
        line=dict(color="MediumPurple"),
        fillcolor="Lavender"
    )
    fig.add_annotation(
        x=motor_x+motor_width/2, y=motor_y+motor_height/2,
        text=f"Motor<br>{motor.power_kW*1.34102:.0f} BHP", 
        showarrow=False, font=dict(size=12), align="center"
    )
    
    # Posição do frame (centralizado)
    frame_x = motor_x + motor_width + 50
    frame_y = canvas_height / 2 - 25
    frame_width = 200
    frame_height = 50
    fig.add_shape(
        type="rect",
        x0=frame_x, y0=frame_y,
        x1=frame_x+frame_width, y1=frame_y+frame_height,
        line=dict(color="RoyalBlue"),
        fillcolor="LightSkyBlue"
    )
    fig.add_annotation(
        x=frame_x+frame_width/2, y=frame_y+frame_height/2,
        text=f"Frame<br>RPM: {frame.rpm:.0f}", showarrow=False, font=dict(size=12), align="center"
    )
    
    # Distribuição dos throws abaixo do frame
    n = len(throws)
    if n > 0:
        throw_spacing = frame_width / n
    else:
        throw_spacing = 0
    for t in throws:
        idx = t.throw_number - 1
        throw_x = frame_x + idx * throw_spacing + throw_spacing/4
        throw_y = frame_y + frame_height + 20
        throw_width = throw_spacing/2
        throw_height = 30
        fig.add_shape(
            type="rect",
            x0=throw_x, y0=throw_y,
            x1=throw_x+throw_width, y1=throw_y+throw_height,
            line=dict(color="DarkOrange"),
            fillcolor="Moccasin"
        )
        fig.add_annotation(
            x=throw_x+throw_width/2, y=throw_y+throw_height/2,
            text=f"Throw {t.throw_number}", showarrow=False, font=dict(size=10)
        )
    
    # Representa o atuador (acionador) à direita do frame
    actuator_x = frame_x + frame_width + 50
    actuator_y = canvas_height / 2 - 20
    fig.add_shape(
        type="rect",
        x0=actuator_x, y0=actuator_y,
        x1=actuator_x+120, y1=actuator_y+60,
        line=dict(color="SaddleBrown"),
        fillcolor="PeachPuff"
    )
    fig.add_annotation(
        x=actuator_x+60, y=actuator_y+30,
        text=f"Acionador<br>{actuator.power_kW:.0f} kW", showarrow=False, font=dict(size=12), align="center"
    )
    
    fig.update_layout(
        width=canvas_width,
        height=canvas_height,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    return fig

# ------------------------------------------------------------------------------
# Interface do usuário com Streamlit
# ------------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Calculadora de Performance de Compressor - Ariel7-style", layout="wide")
    st.title("Calculadora de Performance de Compressor (Estilo Ariel 7)")
    
    init_db()
    
    # Seleção de unidades
    UNIT_OPTIONS = ["SI", "Metric"]
    if "unit_system" not in st.session_state:
        st.session_state["unit_system"] = "SI"
    
    with st.sidebar:
        st.header("Configurações Gerais")
        unit = st.selectbox("Sistema de unidades", options=UNIT_OPTIONS, index=0)
        st.session_state["unit_system"] = unit
        if st.button("Resetar DB"):
            import os
            if os.path.exists("compressor.db"):
                os.remove("compressor.db")
            init_db()
            st.success("Banco de dados reinicializado.")
    
    tabs = st.tabs(["Processo", "Configuração do Equipamento"])
    
    # --- Aba PROCESSO ---
    with tabs[0]:
        st.header("Processo")
        st.subheader("Condições de Processo")
        if st.session_state["unit_system"] == "SI":
            inlet_pressure = st.number_input("Pressão de sucção (Pa)", value=200000.0, step=1000.0)
            inlet_temperature = st.number_input("Temperatura de entrada (K)", value=298.15, step=1.0)
        else:
            inlet_pressure_bar = st.number_input("Pressão de sucção (bar)", value=2.0, step=0.01)
            inlet_temperature_C = st.number_input("Temperatura de entrada (°C)", value=25.0, step=1.0)
            inlet_pressure = inlet_pressure_bar * 1e5
            inlet_temperature = inlet_temperature_C + 273.15
        
        mass_flow = st.number_input("Massa de Gás (kg/s)", value=12.0, step=0.1)
        n_stages = st.number_input("Número de estágios", min_value=1, max_value=12, value=3, step=1)
        PR_total = st.number_input("Razão de compressão total (PR)", min_value=1.0, max_value=100.0, value=2.5, step=0.1)
        
        st.subheader("Calcular Performance")
        if st.button("Calcular outputs (Processo)"):
            # Para cálculo completo, os throws e mapeamento deverão ser configurados na aba de equipamento.
            calc_outputs = perform_performance_calculation(
                mass_flow=mass_flow,
                inlet_pressure=Q_(inlet_pressure, ureg.Pa),
                inlet_temperature=Q_(inlet_temperature, ureg.K),
                n_stages=n_stages,
                PR_total=PR_total,
                throws=[],           # vazia neste caso
                stage_mapping={},
                actuator=Actuator(power_kW=0, derate_percent=0, air_cooler_fraction=0)
            )
            st.json(calc_outputs)
    
    # --- Aba CONFIGURAÇÃO DO EQUIPAMENTO ---
    with tabs[1]:
        st.header("Configuração do Equipamento")
        st.subheader("Frame, Throws e Mapeamento de Estágios")
        # Inputs do Frame
        frame_rpm = st.number_input("RPM do Frame", min_value=100, max_value=3000, value=900, step=10)
        if st.session_state["unit_system"] == "SI":
            stroke_input = st.number_input("Stroke do Frame (m)", min_value=0.01, max_value=1.0, value=0.12, step=0.01)
        else:
            stroke_input = st.number_input("Stroke do Frame (mm)", min_value=10, max_value=1000, value=120, step=1)
        n_throws = st.number_input("Número de Throws", min_value=1, max_value=20, value=3, step=1)
        if st.session_state["unit_system"] == "Metric":
            stroke_si = stroke_input / 1000.0
        else:
            stroke_si = stroke_input
        
        current_frame = Frame(rpm=frame_rpm, stroke=stroke_si, n_throws=n_throws)
        
        st.subheader("Parâmetros dos Throws")
        throws_list = []
        for i in range(1, int(n_throws)+1):
            st.markdown(f"**Throw {i}**")
            if st.session_state["unit_system"] == "SI":
                bore = st.number_input(f"Throw {i} - Bore (m)", value=0.08, min_value=0.01, max_value=0.2, step=0.001, key=f"throw{i}_bore")
                clearance = st.number_input(f"Throw {i} - Clearance (m)", value=0.002, min_value=0.0005, max_value=0.01, step=0.0005, key=f"throw{i}_clearance")
            else:
                bore_mm = st.number_input(f"Throw {i} - Bore (mm)", value=80, min_value=10, max_value=400, step=1, key=f"throw{i}_bore")
                clearance_mm = st.number_input(f"Throw {i} - Clearance (mm)", value=2, min_value=0, max_value=20, step=1, key=f"throw{i}_clearance")
                bore = bore_mm / 1000.0
                clearance = clearance_mm / 1000.0
            VVCP = st.number_input(f"Throw {i} - VVCP (%)", value=90, min_value=0, max_value=100, step=1, key=f"throw{i}_VVCP")
            SACE = st.number_input(f"Throw {i} - SACE (%)", value=80, min_value=0, max_value=100, step=1, key=f"throw{i}_SACE")
            SAHE = st.number_input(f"Throw {i} - SAHE (%)", value=60, min_value=0, max_value=100, step=1, key=f"throw{i}_SAHE")
            throws_list.append(Throw(throw_number=i, bore=bore, clearance=clearance, VVCP=VVCP, SACE=SACE, SAHE=SAHE))
        
        st.subheader("Mapeamento de Estágios para Throws")
        # Permite selecionar mais de um throw para cada estágio
        stage_mapping = {}
        for s in range(1, int(n_stages)+1):
            options = [f"Throw {t.throw_number}" for t in throws_list]
            selected = st.multiselect(f"Estágio {s} recebe:", options=options, key=f"stage_map_{s}")
            # Extrai os números dos throws selecionados
            selected_ids = []
            for sel in selected:
                try:
                    num = int(sel.split(" ")[1])
                    selected_ids.append(num)
                except Exception:
                    pass
            stage_mapping[s] = selected_ids
        
        st.subheader("Parâmetros do Atuador")
        power_kW = st.number_input("Potência do Acionador (kW)", value=250.0, min_value=0.0, step=1.0)
        derate_pct = st.number_input("Derate (%)", value=5.0, min_value=0.0, max_value=100.0, step=0.5)
        air_cooler_frac = st.number_input("Air Cooler (%)", value=25.0, min_value=0.0, max_value=100.0, step=0.5)
        current_actuator = Actuator(power_kW=power_kW, derate_percent=derate_pct, air_cooler_fraction=air_cooler_frac)
        
        st.subheader("Parâmetros do Motor")
        # O motor aqui representa a fonte de potência, normalmente com potência em kW, convertida para BHP para diagrama
        motor_power_kW = st.number_input("Potência do Motor (kW)", value=300.0, min_value=0.0, step=1.0)
        current_motor = Motor(power_kW=motor_power_kW)
        
        st.markdown("---")
        st.subheader("Diagrama do Equipamento")
        fig_diagram = generate_diagram(current_frame, throws_list, current_actuator, current_motor)
        st.plotly_chart(fig_diagram, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Salvar Configuração e Calcular Performance")
        if st.button("Salvar Configuração e Calcular Outputs"):
            db = SessionLocal()
            # Salvar Frame
            frame_model = FrameModel(rpm=frame_rpm, stroke_m=stroke_si, n_throws=n_throws)
            db.add(frame_model)
            db.commit()
            db.refresh(frame_model)
            # Salvar cada Throw
            for t in throws_list:
                throw_model = ThrowModel(
                    frame_id=frame_model.id,
                    throw_number=t.throw_number,
                    bore_m=t.bore,
                    clearance_m=t.clearance,
                    VVCP=t.VVCP,
                    SACE=t.SACE,
                    SAHE=t.SAHE
                )
                db.add(throw_model)
            # Salvar Atuador
            actuator_model = ActuatorModel(
                power_available_kW=power_kW,
                derate_percent=derate_pct,
                air_cooler_fraction=air_cooler_frac
            )
            db.add(actuator_model)
            db.commit()
            db.close()
            
            # Exemplo de cálculo de performance com dados do processo fixos
            calc_outputs = perform_performance_calculation(
                mass_flow=12.0,
                inlet_pressure=Q_(6000000, ureg.Pa),  # 60 bar
                inlet_temperature=Q_(298.15, ureg.K),
                n_stages=n_stages,
                PR_total=2.5,
                throws=throws_list,
                stage_mapping=stage_mapping,
                actuator=current_actuator
            )
            calc_outputs["frame_rpm"] = frame_rpm
            st.success("Configuração salva e outputs calculados:")
            st.json(calc_outputs)

if __name__ == "__main__":
    main()'''
 
# Save app.py inside the project folder
with open(os.path.join(project_folder, 'app.py'), 'w') as f:
    f.write(app_code)

# -------------------------------------------------------------------
# Write requirements.txt
# -------------------------------------------------------------------
requirements = """streamlit
pandas
plotly
pint
sqlalchemy
"""
with open(os.path.join(project_folder, 'requirements.txt'), 'w') as f:
    f.write(requirements)

# -------------------------------------------------------------------
# Write README.md
# -------------------------------------------------------------------
readme = """# Demag Streamlit App

Este repositório contém a aplicação Streamlit para cálculo de performance de compressores, inspirada no Ariel7.

## Conteúdo
- **app.py**: Código principal da aplicação.
- **requirements.txt**: Dependências para execução.
- **README.md**: Este arquivo.
"""
with open(os.path.join(project_folder, 'README.md'), 'w') as f:
    f.write(readme)

# -------------------------------------------------------------------
# Create a zip archive of the project folder
# -------------------------------------------------------------------
with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(project_folder):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, project_folder)
            zipf.write(file_path, arcname)

print(f"Zip archive created at: {zip_file_path}")
'''

# -------------------------------------------------------------------
# End of the script.
# -------------------------------------------------------------------
