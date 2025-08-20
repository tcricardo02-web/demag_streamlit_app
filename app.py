# … your existing imports, logger, pint, SQLAlchemy setup, data‐classes, etc. …

# --- New motor‐curve helper ----------------------------------
from scipy.interpolate import interp1d

@dataclass
class MotorCurve:
    """ Holds discrete (rpm→power_kW) points, 
        and provides interp at any RPM. """
    rpm_points: List[float]
    power_kW:   List[float]
    kind: str = 'linear'
    def __post_init__(self):
        # create monotonic interpolation
        self._f = interp1d(self.rpm_points, self.power_kW,
                           kind=self.kind, fill_value="extrapolate")
    def available_power(self, rpm: float) -> float:
        return float(self._f(rpm))

# --- EXTEND perform_performance_calculation ------------------
def perform_performance_calculation(
    mass_flow, P_in, T_in, P_out,
    throws: List[Throw],
    actuator: Actuator,
    motor_curve: MotorCurve,
    n_stages: int,
):
    """ Now takes explicit n_stages and motor_curve """
    # 1) get motor‐available kW & derate:
    avail_kw = motor_curve.available_power(motor_curve.current_rpm)
    avail_kw *= (1 - actuator.derate_percent/100)
    # (you may choose to cap total_W_kW <= avail_kw)
    # … then same stage loop as before, BUT:
    details = []
    P = P_in.to(ureg.Pa).magnitude
    T = T_in.to(ureg.K).magnitude
    gamma, cp = 1.30, 2.0
    PR_total = P_out.to(ureg.Pa).magnitude / P
    PR_base  = PR_total ** (1/n_stages)
    total_kW = 0.0

    for stage in range(1, n_stages+1):
        P_in_s = P * (PR_base**(stage-1))
        P_out_s= P_in_s * PR_base

        # average VVCP/SACE/clearance only among throws assigned to this stage
        assigned = [t for t in throws if t.stage_assignment==stage]
        vvcp = np.mean([t.VVCP_pct for t in assigned]) if assigned else 0.
        clr  = np.mean([t.clearance_pct for t in assigned]) if assigned else 0.
        # efficiency model (unchanged)
        eta = clamp(0.65 +0.15*(vvcp/100) -0.05*(vvcp/100) +0.10*(clr/100), 0.65,0.92)

        # isentropic temperature
        T_is = T*(PR_base**((gamma-1)/gamma))
        Tout = T + (T_is - T)/eta
        dT   = Tout - T
        Wk   = mass_flow * cp * dT / 1000
        total_kW += Wk
        # record
        details.append({ 
            'stage': stage,
            'P_in_bar':  P_in_s/1e5,
            'T_in_C':    T-273.15,
            'P_out_bar': P_out_s/1e5,
            'T_out_C':   Tout-273.15,
            'eff':       eta,
            'W_kW':      Wk
        })
        # now interstage cooler:
        # drop 1% pressure, reset temperature to 120 °F
        P = P_out_s*0.99
        Tout = (120-32)*5/9 + 273.15  # 120 °F → K
        T = Tout

    total_BHP = total_kW * 1.34102
    return {
      'total_kW': total_kW,
      'total_BHP': total_BHP,
      'details': details
    }


# --- Streamlit UI --------------------------------------------
def main():
    st.set_page_config("Ariel7 Compressor", "wide")
    init_db()

    # Persisted state for curve‐points
    if 'motor_curve_pts' not in st.session_state:
        # default example: two points
        st.session_state.motor_curve_pts = {'rpm':[900,1200],'kW':[200,300]}

    tabs = st.tabs(["Processo","Equipamento","Report","Multi-Run"])

    # — Equipment Tab ------------------------------------------
    with tabs[1]:
        st.header("Configuração do Compressor")

        # 1) Stage & frame
        n_stages = st.number_input("Número de Estágios", min_value=1, value=3, step=1)
        rpm       = st.number_input("Frame RPM", value=900, step=10)
        stroke    = st.number_input("Stroke (m)", value=0.12, format="%.3f")

        n_throws  = st.number_input("Número de Throws", min_value=1, value=3, step=1)
        throws: List[Throw] = []
        for i in range(1, n_throws+1):
            st.markdown(f"Throw {i}")
            stage_assign = st.selectbox(
                f"  Estágio p/Throw {i}", 
                options=list(range(1,n_stages+1)),
                key=f"stg_{i}"
            )
            vvcp_pct  = st.slider(f"  VVCP % #{i}",  0.0,100.0, 90.0, key=f"v_{i}")
            clr_pct   = st.slider(f"  Clearance %#{i}",0.0,100.0,  2.0, key=f"c_{i}")
            throws.append(Throw(
                throw_number=i,
                stage_assignment=stage_assign,
                bore=0, clearance=0,
                VVCP_pct=vvcp_pct,
                clearance_pct=clr_pct,
                SACE=0, SAHE=0
            ))

        # 2) Actuator
        pw_avail = st.number_input("Atuador kW Nominal", value=250.0)
        derate   = st.number_input("Derate %", value=5.0)
        ac_frac  = st.number_input("Air-Cooler %", value=25.0)
        actuator = Actuator(pw_avail, derate, ac_frac)

        # 3) Motor type & curve
        motor_type = st.radio("Tipo de Motor", ["Elétrico","Gás Natural"])
        st.markdown("## Motor Power Curve (RPM → kW)")
        df_curve = pd.DataFrame(st.session_state.motor_curve_pts)
        edited  = st.data_editor(df_curve, num_rows="dynamic")
        # save back
        st.session_state.motor_curve_pts = {
            'rpm': list(edited['rpm']),
            'kW':  list(edited['kW'])
        }
        motor_curve = MotorCurve(
            rpm_points=st.session_state.motor_curve_pts['rpm'],
            power_kW=  st.session_state.motor_curve_pts['kW']
        )
        motor_curve.current_rpm = rpm

        # store all config
        st.session_state.eq_config = dict(
            n_stages=n_stages, rpm=rpm, stroke=stroke,
            throws=throws, actuator=actuator, motor_curve=motor_curve
        )

        st.success("Configuração salva.")

    # — Process Tab --------------------------------------------
    with tabs[0]:
        st.header("Processo e Diagrama")
        if 'process' not in st.session_state or st.button("Carregar Processo"):
            # gather inlet/outlet inputs
            pin = st.number_input("P_sucção psig", 30.0)*6894.76
            tin = (st.number_input("T_sucção °F",77.0)-32)*5/9+273.15
            pout= st.number_input("P_descarga psig",60.0)*6894.76
            mf  = st.number_input("Fluxo kg/s",12.0)
            st.session_state.process = dict(
                P_in=Q_(pin,ureg.Pa),
                T_in=Q_(tin,ureg.K),
                P_out=Q_(pout,ureg.Pa),
                mf=mf
            )
        if 'eq_config' in st.session_state and 'process' in st.session_state:
            cfg = st.session_state.eq_config
            pr = st.session_state.process
            out=perform_performance_calculation(
                pr['mf'], pr['P_in'], pr['T_in'], pr['P_out'],
                cfg['throws'], cfg['actuator'], cfg['motor_curve'],
                cfg['n_stages']
            )

            # Draw P–T diagram with Plotly
            fig = go.Figure()
            for row in out['details']:
                # plot compressor stage as an arrow
                fig.add_trace(go.Scatter(
                  x=[row['P_in_bar'], row['P_out_bar']],
                  y=[row['T_in_C'],   row['T_out_C']],
                  mode='lines+markers',
                  name=f"Stg {row['stage']}"
                ))
                # plot cooler drop (vertical T-drop at constant P)
                fig.add_trace(go.Scatter(
                  x=[row['P_out_bar'], row['P_out_bar']*0.99/1e5],
                  y=[row['T_out_C'], 120.0],  # drop to 120 °F=48.9 °C 
                  mode='lines',
                  line=dict(dash='dash'),
                  showlegend=False
                ))
            fig.update_layout(
              xaxis_title="Pressure (bar)",
              yaxis_title="Temperature (°C)",
              title="Process P–T Diagram"
            )
            st.plotly_chart(fig, use_container_width=True)

    # — Multi-Run Tab ------------------------------------------
    with tabs[3]:
        st.header("Multi-Run Sweep")
        cfg = st.session_state.get('eq_config',{})
        pr  = st.session_state.get('process',{})
        c1,c2 = st.columns(2)
        Pmin = c1.number_input("P_out min psig", value=40.0)
        Pmax = c1.number_input("P_out max psig", value=100.0)
        dP   = c1.number_input("ΔP step", value=5.0)
        rpm_min = c2.number_input("RPM min", 600)
        rpm_max = c2.number_input("RPM max",1200)
        dRPM    = c2.number_input("ΔRPM",    100)

        if st.button("Executar Multi-Run"):
            rows=[]
            for P in np.arange(Pmin,Pmax+dP/2,dP):
                pout_loop=Q_(P*6894.76,ureg.Pa)
                for R in np.arange(rpm_min,rpm_max+dRPM/2,dRPM):
                    cfg['motor_curve'].current_rpm = R
                    out = perform_performance_calculation(
                      pr['mf'], pr['P_in'], pr['T_in'], pout_loop,
                      cfg['throws'], cfg['actuator'],
                      cfg['motor_curve'], cfg['n_stages']
                    )
                    rows.append({
                      'P_out_psig':P,
                      'RPM':R,
                      'Flow':pr['mf'],
                      'BHP':out['total_BHP']
                    })
            dfm = pd.DataFrame(rows)
            fig1=px.line(dfm, x='P_out_psig',y='Flow',color='RPM',markers=True,
                         title="Flow vs P_out")
            fig2=px.line(dfm, x='P_out_psig',y='BHP', color='RPM',markers=True,
                         title="BHP vs P_out")
            st.plotly_chart(fig1,use_container_width=True)
            st.plotly_chart(fig2,use_container_width=True)

    # — Report Tab remains largely the same…  
    #    you can save out to CSV/DB as before.

if __name__=='__main__':
    main()
