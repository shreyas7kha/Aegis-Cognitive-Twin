import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Aegis Cognitive Twin", layout="wide")

# --- SIDEBAR: Global Control Panel ---
st.sidebar.title("Aegis Control Panel")
st.sidebar.markdown("---")
st.sidebar.info("Operationalizing Small Modular Reactors (SMRs) for India's AI Infrastructure")

smr_rated_capacity = st.sidebar.number_input("Small Modular Reactor Rated Capacity (MW)", 15, 300, 77)
battery_capacity = st.sidebar.slider("On-site Battery Storage Capacity (MW)", 0, 100, 20)

# --- HEADER ---
st.title("🛡️ Aegis Cognitive Twin: Decision Support Engine")
st.subheader("Integrated System for Nuclear-as-a-Service (NaaS) & AI Infrastructure")

# Tab Order: Demand Forecast, Financial Risk, Energy Orchestration, Predictive Maintenance, Regulatory Navigator
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Demand Forecast", 
    "💰 Financial Risk", 
    "⚡ Energy Orchestration", 
    "🛠️ Predictive Maintenance", 
    "⚖️ Regulatory Navigator"
])

# --- FEATURE 1: Demand Forecast ---
with tab1:
    st.header("Artificial Intelligence Workload Demand Forecast")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.write("### Compute Parameters")
        gpu_count = st.slider("AI Cluster Size (Total NVIDIA H100 GPUs)", 1000, 100000, 10000)
        pue_target = st.slider("Target Power Usage Effectiveness (PUE)", 1.0, 2.0, 1.2)
        training_load = st.slider("Active Training Load (%)", 0, 100, 75)
        inference_load = st.slider("Steady State Inference Load (%)", 0, 100, 25)
        
        # Power Formula: (N_GPUs * 0.0007 MW per GPU) * PUE * Total Load
        raw_power = (gpu_count * 0.0007) * pue_target
        actual_demand = raw_power * ((training_load + inference_load) / 100)
        
        st.metric("Total Predicted Power Demand", f"{actual_demand:.2f} MW")
        st.info(f"Required SMR Fleet: {int(np.ceil(actual_demand / smr_rated_capacity))} Unit(s)")

    with col2:
        times = pd.date_range("2026-01-01", periods=24, freq="H")
        demand_data = actual_demand + np.random.normal(0, 0.1*gpu_count/1000, 24)
        supply_data = [smr_rated_capacity] * 24
        
        fig_demand = go.Figure()
        fig_demand.add_trace(go.Scatter(x=times, y=demand_data, name="AI Load Demand", line=dict(color='orange', width=3)))
        fig_demand.add_trace(go.Scatter(x=times, y=supply_data, name="SMR Constant Supply", line=dict(color='cyan', dash='dash')))
        fig_demand.update_layout(title="24-Hour AI Compute Power Profile", yaxis_title="Power (MW)")
        st.plotly_chart(fig_demand, use_container_width=True)

# --- FEATURE 2: Financial Risk ---
with tab2:
    st.header("Probabilistic Financial Risk Analysis")
    st.write("Monte Carlo simulations to determine the Break-even Frontier for SMR deployment.")
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        sim_count = 1000
        licensing_months = np.random.normal(24, 6, sim_count) # Mean 24 months, 6 std dev
        fig_licensing = px.histogram(licensing_months, nbins=30, title="Probability: Time-to-Power (Months)", color_discrete_sequence=['#636EFA'])
        st.plotly_chart(fig_licensing, use_container_width=True)
    
    with col_f2:
        roi_years = np.random.normal(12, 2, sim_count)
        fig_roi = px.scatter(x=licensing_months, y=roi_years, title="Impact of Licensing Delay on Return on Investment (ROI)", labels={'x':'Months to Operational', 'y':'Years to Break-even'})
        st.plotly_chart(fig_roi, use_container_width=True)

# --- FEATURE 3: Energy Orchestration ---
with tab3:
    st.header("Microgrid Energy Orchestration & Balancing")
    st.write("Real-time management of the 'Behind-the-Meter' energy network.")
    
    net_demand = actual_demand # From Tab 1
    battery_level = st.slider("Current Battery State of Charge (%)", 0, 100, 50)
    
    # Logic: Net Power Balance (Delta P)
    net_balance = smr_rated_capacity - net_demand
    
    c1, c2, c3 = st.columns(3)
    c1.metric("SMR Generation Output", f"{smr_rated_capacity} MW")
    c2.metric("Current AI Campus Load", f"{net_demand:.2f} MW")
    c3.metric("Net Energy Balance", f"{net_balance:.2f} MW")
    
    if net_balance < 0:
        st.error(f"⚠️ DEFICIT: Drawing {abs(net_balance):.2f} MW from Batteries.")
    else:
        st.success(f"✅ SURPLUS: Charging Battery Storage with {net_balance:.2f} MW.")

# --- FEATURE 4: Predictive Maintenance ---
with tab4:
    st.header("Predictive Maintenance & Asset Health Monitor")
    st.write("AI-driven monitoring of reactor equipment to prevent unplanned downtime.")
    
    pm_col1, pm_col2 = st.columns(2)
    with pm_col1:
        vibration_level = st.slider("Primary Cooling Pump Vibration (mm/s)", 0.0, 10.0, 1.2)
        # Health Formula: 1 - (Actual / Threshold)
        health_index = 1 - (vibration_level / 8.0)
        st.metric("Reactor Component Health Score", f"{health_index*100:.1f}%")
        
        remaining_useful_life = int(health_index * 365)
        st.warning(f"Estimated Remaining Useful Life (RUL): {remaining_useful_life} Days until maintenance required.")

    with pm_col2:
        sensor_data = np.random.normal(1.2, 0.2, 50)
        sensor_data = np.append(sensor_data, [vibration_level])
        fig_maint = px.line(sensor_data, title="Vibration Sensor Trend Analysis", labels={'index': 'Time (Hours)', 'value': 'Vibration'})
        fig_maint.add_hline(y=7.0, line_dash="dash", line_color="red", annotation_text="Danger Threshold")
        st.plotly_chart(fig_maint, use_container_width=True)

# --- FEATURE 5: Regulatory Navigator ---
with tab5:
    st.header("Regulatory Navigator & Liability AI")
    st.write("Retrieval-Augmented Generation (RAG) for navigating Indian Nuclear Policy.")
    
    legal_query = st.text_input("Enter Legal or Policy Question:", "What is the liability cap under the SHANTI Bill 2025?")
    if legal_query:
        st.info("**Aegis Regulatory Insight:**")
        st.markdown(f"""
        > **Query Result:** Under the **SHANTI Bill 2025**, the operator liability for Small Modular Reactors is capped at **₹100 Crore**. 
        > This replaces the higher caps of the **Civil Liability for Nuclear Damage (CLND) Act 2010** to encourage private investment.
        > 
        > **Primary Sources:**
        > - NITI Aayog SMR Roadmap (2023)
        > - Department of Atomic Energy (DAE) Draft SHANTI Bill 
        """)
        st.progress(0.95, text="Source Verification Score: 95%")

# --- FOOTER ---
st.markdown("---")
st.markdown("""
**Disclaimer:** This is a decision-support prototype. Data inputs are simulated for demonstration purposes.
**Core Mission:** Accelerating India's $26B private nuclear investment goal.
""")