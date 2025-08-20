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
        # simple linear interpolation (and extrapolation)
        return float(np.interp(rpm, self.rpm_points, self.power_kW))

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
    # 1) Motor available power (derated)
    avail_kw = motor_curve.available_power(motor_curve.current_rpm)
    avail_kw *= (1 - actuator.derate_percent / 100)

    # 2) Thermodynamic compression + interstage cooling
    P = P_in.to(ureg.Pa).magnitude
    T = T_in.to(ureg.K).magnitude
    gamma, cp = 1.30, 2.0

    PR_total = P_out.to(ureg.Pa).magnitude / P
    PR_base  = PR_total ** (1 / n_stages)

    total_kW = 0.0
    details  = []

    for stage in range(1, n_stages + 1):
        Pin_s  = P * (PR_base ** (stage - 1))
        Pout_s = Pin_s * PR_base

        assigned = [t for t in throws if t.stage_assignment == stage]
        vvcp = np.mean([t.VVCP_pct for t in assigned]) if assigned else 0.0
        clr  = np.mean([t.clearance_pct for t in assigned]) if assigned else 0.0

        eta = clamp(
            0.65
            + 0.15 * (vvcp / 100)
            - 0.05 * (vvcp / 100)
            + 0.10 * (clr / 100),
            0.65,
            0.92
        )

        T_is = T * (PR_base ** ((gamma - 1) / gamma))
        Tout = T
