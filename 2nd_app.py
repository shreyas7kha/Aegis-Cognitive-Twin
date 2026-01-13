import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Aegis Cognitive Twin", layout="wide")

# -----------------------------
# GLOBAL CONSTANTS / HELPERS
# -----------------------------
np.random.seed(7)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def fmt_range(p50, p90, unit=""):
    return f"{p50:.0f}{unit}–{p90:.0f}{unit}"

def percentile(x, p):
    return float(np.percentile(x, p))

@dataclass
class DecisionOutputs:
    viability: str
    binding_constraint: str
    time_to_power_p50: float
    time_to_power_p90: float
    break_even_p50_years: float
    break_even_p90_years: float
    confidence: str
    recommendation: str

def build_decision(
    supply_mw: float,
    demand_profile: np.ndarray,
    licensing_months_samples: np.ndarray,
    break_even_years_samples: np.ndarray,
    legal_coverage_score: float,
    availability_risk_score: float,
):
    """
    Investor-style decision logic (simple, explainable, defensible).
    You can refine weights later; this is prototype-grade but coherent.
    """
    peak_demand = float(np.max(demand_profile))
    avg_demand = float(np.mean(demand_profile))

    # Baseload adequacy: SMR treated as fixed output; flexibility downstream
    # We consider "adequate" if supply >= peak demand OR if supply >= avg and storage can buffer peaks.
    # For this prototype, we use a soft threshold.
    supply_ratio_peak = supply_mw / max(peak_demand, 1e-6)
    supply_ratio_avg = supply_mw / max(avg_demand, 1e-6)

    # Licensing uncertainty
    ttp_p50 = percentile(licensing_months_samples, 50)
    ttp_p90 = percentile(licensing_months_samples, 90)

    # Economics uncertainty
    be_p50 = percentile(break_even_years_samples, 50)
    be_p90 = percentile(break_even_years_samples, 90)

    # Decision heuristics (transparent)
    # - Viable if:
    #   (a) supply adequacy is decent AND (b) time-to-power not extreme AND (c) break-even reasonable
    # - “Conditional” if one pillar is borderline.
    # - “Not viable” if multiple pillars fail.

    supply_flag = (supply_ratio_peak >= 1.0) or (supply_ratio_avg >= 0.9)
    ttp_flag = (ttp_p90 <= 42)  # months; long-tail beyond ~3.5 years becomes hard for AI capacity bets
    econ_flag = (be_p90 <= 18)  # years; depends on buyer patience & discount rate
    reg_flag = (legal_coverage_score >= 0.7)
    avail_flag = (availability_risk_score <= 0.35)  # lower is better

    flags = {
        "Supply adequacy": supply_flag,
        "Time-to-power": ttp_flag,
        "Economics": econ_flag,
        "Regulatory coverage": reg_flag,
        "Availability risk": avail_flag
    }

    failed = [k for k, v in flags.items() if not v]

    # Binding constraint: pick the "weakest" by severity
    # Simple severity scores
    sev = {
        "Supply adequacy": (0 if supply_flag else (1.0 - supply_ratio_avg)),
        "Time-to-power": (0 if ttp_flag else (ttp_p90 - 42) / 24),
        "Economics": (0 if econ_flag else (be_p90 - 18) / 10),
        "Regulatory coverage": (0 if reg_flag else (0.7 - legal_coverage_score)),
        "Availability risk": (0 if avail_flag else (availability_risk_score - 0.35))
    }
    binding = max(sev, key=sev.get)
    binding_constraint = binding

    # Confidence (prototype): depends on reg coverage + availability signal quality
    conf_score = 0.45 * legal_coverage_score + 0.55 * (1.0 - availability_risk_score)
    if conf_score >= 0.78:
        confidence = "High"
    elif conf_score >= 0.62:
        confidence = "Medium"
    else:
        confidence = "Low"

    if len(failed) == 0:
        viability = "Viable"
        recommendation = "Proceed to pilot site assessment + term-sheet design."
    elif len(failed) <= 2:
        viability = "Conditional"
        recommendation = f"Proceed only if you mitigate: {', '.join(failed[:2])}."
    else:
        viability = "Not viable"
        recommendation = f"Do not proceed until you resolve: {', '.join(failed[:3])}."

    return DecisionOutputs(
        viability=viability,
        binding_constraint=binding_constraint,
        time_to_power_p50=ttp_p50,
        time_to_power_p90=ttp_p90,
        break_even_p50_years=be_p50,
        break_even_p90_years=be_p90,
        confidence=confidence,
        recommendation=recommendation
    )

# -----------------------------
# SIDEBAR: GLOBAL CONTROLS
# -----------------------------
st.sidebar.title("Aegis Control Panel")
st.sidebar.markdown("---")
st.sidebar.info("Decision-support prototype for Nuclear-as-a-Service (NaaS) + AI data-centre planning.\n\n"
                "Boundary: **Aegis advises; it does not control nuclear assets.**")

smr_rated_capacity = st.sidebar.number_input("SMR Rated Capacity (MW)", 15, 300, 77, step=1)
battery_power_capacity = st.sidebar.slider("Battery Power Capacity (MW)", 0.0, 200.0, 20.0, 1.0)
battery_energy_hours = st.sidebar.slider("Battery Energy Duration (hours at rated power)", 0.5, 12.0, 2.0, 0.5)
thermal_storage_mw = st.sidebar.slider("Thermal Storage Absorption (MW equiv.)", 0.0, 200.0, 15.0, 1.0)

st.sidebar.markdown("---")
st.sidebar.subheader("Economics & Risk Inputs")
downtime_cost_per_hr = st.sidebar.number_input("Downtime Cost (₹ crore / hour)", 0.1, 50.0, 5.0, step=0.1)
discount_rate = st.sidebar.slider("Discount Rate (%)", 3.0, 18.0, 10.0, 0.5) / 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("Regulatory Coverage (Prototype)")
legal_coverage_score = st.sidebar.slider("Source Coverage Score (0–1)", 0.0, 1.0, 0.85, 0.01,
                                        help="Prototype proxy for completeness of retrieved primary sources in the RAG module.")

st.sidebar.markdown("---")
st.sidebar.subheader("Model Assumptions")
show_assumptions = st.sidebar.checkbox("Show assumptions panels", value=True)

# -----------------------------
# HEADER
# -----------------------------
st.title("🛡️ Aegis Cognitive Twin: Decision Support Engine")
st.subheader("AI-native platform to evaluate SMR co-location for AI data centres under uncertainty")

# -----------------------------
# TABS
# -----------------------------
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "✅ Executive Summary",
    "📈 Workload & Facility Twin",
    "💰 Probabilistic Finance",
    "⚡ Energy & Thermal Routing",
    "🧭 Asset Availability Intelligence",
    "⚖️ Regulatory Navigator (RAG)"
])

# -----------------------------
# TAB 1: Workload & Facility Twin
# -----------------------------
with tab1:
    st.header("Workload–Facility Intelligence (AI Demand + Cooling)")
    left, right = st.columns([1, 2])

    with left:
        st.write("### Scenario Presets (Investor-demo friendly)")
        scenario = st.selectbox(
            "Select workload regime",
            [
                "Inference-dominant steady state",
                "Fine-tuning sprint (bursty)",
                "Pre-training week (high utilization)",
                "Mixed enterprise workload (spiky)"
            ]
        )

        st.write("### Compute & Facility Parameters")
        gpu_count = st.slider("AI Cluster Size (NVIDIA H100 GPUs)", 1000, 200000, 10000, step=1000)
        pue_target = st.slider("Target PUE", 1.0, 2.0, 1.2, 0.05)
        cooling_fraction = st.slider("Cooling share of total facility load (%)", 10, 60, 30, 1)

        # Regime parameters (simple but explainable)
        if scenario == "Inference-dominant steady state":
            base_util = st.slider("Average utilization (%)", 10, 60, 25, 1)
            peakiness = 0.15
            burst_prob = 0.08
        elif scenario == "Fine-tuning sprint (bursty)":
            base_util = st.slider("Average utilization (%)", 30, 85, 60, 1)
            peakiness = 0.35
            burst_prob = 0.22
        elif scenario == "Pre-training week (high utilization)":
            base_util = st.slider("Average utilization (%)", 60, 95, 85, 1)
            peakiness = 0.20
            burst_prob = 0.12
        else:  # Mixed enterprise workload (spiky)
            base_util = st.slider("Average utilization (%)", 20, 80, 50, 1)
            peakiness = 0.45
            burst_prob = 0.25

        # Simple, transparent power model:
        # - Approx H100 power ~0.7 kW average (varies by workload); we use 0.7 kW baseline.
        # - Convert to MW; apply utilization; then apply PUE to get facility power.
        # This is a prototype assumption; put it in "Assumptions".
        gpu_kw = 0.7
        it_power_mw = (gpu_count * gpu_kw) / 1000.0  # MW at 100% utilization
        avg_demand_mw = it_power_mw * (base_util / 100.0) * pue_target

        st.metric("Average Predicted Facility Demand", f"{avg_demand_mw:.2f} MW")

        # Fleet sizing guidance
        required_units = int(np.ceil(avg_demand_mw / max(smr_rated_capacity, 1e-6)))
        st.info(f"Indicative SMR Fleet (avg basis): **{required_units} unit(s)**")

        if show_assumptions:
            with st.expander("Model assumptions (transparent)"):
                st.markdown(
                    "- GPU power proxy: **0.7 kW per H100** (order-of-magnitude placeholder; varies by workload).\n"
                    "- Facility demand = IT power × utilization × PUE.\n"
                    "- Workload volatility is simulated (bursts + noise) for demonstration.\n"
                    "- Output is decision-support only; validate with site engineering studies."
                )

    # Build 24h demand profile (volatility driven by regime)
    times = pd.date_range("2026-01-01", periods=24, freq="H")
    demand = []
    for _ in range(24):
        burst = np.random.rand() < burst_prob
        mult = 1.0 + (peakiness * (0.8 + 0.4*np.random.rand()) if burst else 0.0)
        noise = np.random.normal(0, 0.03)  # small noise
        demand.append(avg_demand_mw * (mult + noise))
    demand = np.array([max(0.0, d) for d in demand])

    # Store for other tabs
    st.session_state["demand_profile_mw"] = demand
    st.session_state["avg_demand_mw"] = float(np.mean(demand))

    with right:
        supply = np.full(24, smr_rated_capacity)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=demand, name="Facility demand (MW)", line=dict(width=3)))
        fig.add_trace(go.Scatter(x=times, y=supply, name="SMR baseload supply (MW)", line=dict(dash="dash", width=3)))
        fig.update_layout(
            title="24-hour Demand vs SMR Baseload (Aegis assumes SMR output is fixed baseload)",
            yaxis_title="Power (MW)",
            xaxis_title="Time"
        )
        st.plotly_chart(fig, use_container_width=True)

        peak = float(np.max(demand))
        st.caption(f"Peak demand (simulated): **{peak:.2f} MW** | Supply ratio (peak): **{smr_rated_capacity/max(peak,1e-6):.2f}×**")

# -----------------------------
# TAB 2: Probabilistic Finance
# -----------------------------
with tab2:
    st.header("Probabilistic Finance: Time-to-Power + Break-even under Uncertainty")
    st.write("Monte Carlo simulation links licensing delays, capex uncertainty, and downtime exposure to break-even outcomes.")

    # Pull demand from session; fallback if not visited tab1
    demand_profile = st.session_state.get("demand_profile_mw", np.full(24, 10.0))
    avg_demand_mw = st.session_state.get("avg_demand_mw", float(np.mean(demand_profile)))

    colA, colB = st.columns([1, 1])

    # Inputs (kept simple but defensible)
    with colA:
        st.subheader("Simulation knobs")
        sim_count = st.slider("Simulation runs", 200, 5000, 1500, 100)
        licensing_mean = st.slider("Licensing time mean (months)", 6, 60, 24, 1)
        licensing_std = st.slider("Licensing time std dev (months)", 1, 24, 6, 1)

        capex_per_mw = st.number_input("All-in CAPEX (₹ crore / MW)", 1.0, 100.0, 12.0, 0.5)
        opex_per_mwh = st.number_input("OPEX (₹ / kWh)", 0.5, 20.0, 5.0, 0.5)  # placeholder
        tariff_per_kwh = st.number_input("Implied value of electricity avoided (₹ / kWh)", 1.0, 30.0, 9.0, 0.5)

        if show_assumptions:
            with st.expander("Assumptions & interpretation"):
                st.markdown(
                    "- Licensing time modeled as Normal(mean, std) (prototype).\n"
                    "- Break-even combines: (CAPEX, OPEX, implied avoided cost, downtime exposure).\n"
                    "- Downtime cost is user-entered in sidebar to reflect AI-criticality.\n"
                    "- This is a planning lens, not project finance advice."
                )

    # Run simulation
    licensing_months = np.random.normal(licensing_mean, licensing_std, sim_count)
    licensing_months = np.clip(licensing_months, 3, 84)

    # Simple project size: assume one SMR unit for now in finance module (you can extend to fleet)
    project_mw = smr_rated_capacity

    # Annual energy served (MWh) assuming baseload availability ~ 92% (placeholder) and min(supply, demand avg) usage
    # We incorporate demand by assuming offtake is at least avg_demand up to supply.
    utilization_mw = min(project_mw, avg_demand_mw)
    availability = 0.92
    annual_mwh = utilization_mw * 8760 * availability

    # Annual margin (₹ crore)
    # Convert kWh to ₹: (tariff - opex) ₹/kWh * kWh; then to crore.
    annual_kwh = annual_mwh * 1000
    annual_margin_rs = (tariff_per_kwh - opex_per_mwh) * annual_kwh
    annual_margin_crore = annual_margin_rs / 1e7  # 1 crore = 1e7 INR

    # CAPEX total (₹ crore) with uncertainty
    capex_uncertainty = np.random.normal(1.0, 0.15, sim_count)
    capex_uncertainty = np.clip(capex_uncertainty, 0.7, 1.5)
    capex_total = capex_per_mw * project_mw * capex_uncertainty

    # Downtime exposure proxy: assume some unplanned downtime hours per year depends on availability risk (handled in tab4)
    # Here, we use a base of 10–40 hrs/y uncertain.
    unplanned_hrs = np.random.uniform(10, 40, sim_count)
    downtime_cost_crore = unplanned_hrs * downtime_cost_per_hr

    # Effective annual benefit net of downtime penalties
    effective_annual_benefit = np.maximum(0.1, annual_margin_crore - downtime_cost_crore)

    # Break-even years: (capex + delay penalty) / annual benefit
    # Delay penalty: opportunity cost of waiting licensing months (use discount rate)
    delay_years = licensing_months / 12.0
    delay_penalty_factor = (1 + discount_rate) ** delay_years
    break_even_years = (capex_total * delay_penalty_factor) / effective_annual_benefit
    break_even_years = np.clip(break_even_years, 1.0, 60.0)

    # Store for executive summary
    st.session_state["licensing_months_samples"] = licensing_months
    st.session_state["break_even_years_samples"] = break_even_years

    with colB:
        st.subheader("Outputs")
        p50_ttp = percentile(licensing_months, 50)
        p90_ttp = percentile(licensing_months, 90)
        p50_be = percentile(break_even_years, 50)
        p90_be = percentile(break_even_years, 90)

        c1, c2 = st.columns(2)
        c1.metric("Time-to-Power (P50–P90)", f"{fmt_range(p50_ttp, p90_ttp, ' mo')}")
        c2.metric("Break-even (P50–P90)", f"{fmt_range(p50_be, p90_be, ' yrs')}")

        fig1 = px.histogram(licensing_months, nbins=30, title="Time-to-Power Distribution (months)")
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.scatter(
            x=licensing_months,
            y=break_even_years,
            title="Delay vs Break-even (uncertainty cloud)",
            labels={"x": "Months to operational", "y": "Years to break-even"}
        )
        st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# TAB 3: Energy & Thermal Routing
# -----------------------------
with tab3:
    st.header("Energy & Thermal Routing Advisor (Downstream Flexibility)")
    st.info("Assumption: **SMR output is fixed baseload**. Aegis advises how downstream systems (battery/thermal/cooling) absorb variability.")

    demand_profile = st.session_state.get("demand_profile_mw", np.full(24, 10.0))
    current_load = float(demand_profile[-1])
    supply = smr_rated_capacity

    # Simple routing:
    # - If surplus: charge battery up to power cap; remainder -> thermal storage up to cap; rest curtailed
    # - If deficit: discharge battery up to cap; remainder -> shortfall (flag)
    soc = st.slider("Battery State of Charge (%)", 0, 100, 50, 1)
    soc_frac = soc / 100.0

    battery_energy_mwh = battery_power_capacity * battery_energy_hours
    available_discharge_mwh = battery_energy_mwh * soc_frac
    available_discharge_mw = min(battery_power_capacity, available_discharge_mwh)  # 1-hour decision slice proxy

    net = supply - current_load

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SMR Baseload (MW)", f"{supply:.2f}")
    c2.metric("Current Facility Load (MW)", f"{current_load:.2f}")
    c3.metric("Battery Power Cap (MW)", f"{battery_power_capacity:.2f}")
    c4.metric("Thermal Absorb Cap (MW)", f"{thermal_storage_mw:.2f}")

    if net >= 0:
        to_battery = min(net, battery_power_capacity)
        remaining = net - to_battery
        to_thermal = min(remaining, thermal_storage_mw)
        curtailed = max(0.0, remaining - to_thermal)

        st.success(f"✅ SURPLUS {net:.2f} MW: route **{to_battery:.2f} MW → battery**, **{to_thermal:.2f} MW → thermal**, **{curtailed:.2f} MW curtailed**.")
    else:
        deficit = abs(net)
        from_battery = min(deficit, available_discharge_mw)
        shortfall = max(0.0, deficit - from_battery)
        if shortfall > 0:
            st.error(f"⚠️ DEFICIT {deficit:.2f} MW: **{from_battery:.2f} MW from battery**, **{shortfall:.2f} MW unserved load** (risk).")
        else:
            st.warning(f"🟡 DEFICIT {deficit:.2f} MW: covered by battery discharge **{from_battery:.2f} MW**.")

    if show_assumptions:
        with st.expander("Assumptions (routing)"):
            st.markdown(
                "- Routing is advisory (no control signals).\n"
                "- Battery discharge modeled as a 1-hour slice proxy.\n"
                "- Thermal storage treated as MW-equivalent sink/source for cooling load smoothing.\n"
                "- A full microgrid model (power flow, constraints) is beyond prototype scope."
            )

# -----------------------------
# TAB 4: Asset Availability Intelligence (SAFE REFRAMING)
# -----------------------------
with tab4:
    st.header("Asset Availability Intelligence (Indicative Telemetry → Risk Signals)")
    st.warning("Boundary: Aegis **does not** diagnose faults or prescribe maintenance. It flags indicative availability risks for operator review.")

    left, right = st.columns(2)

    with left:
        st.subheader("Operator-provided telemetry (prototype inputs)")
        vib = st.slider("Pump vibration (mm/s)", 0.0, 10.0, 1.2, 0.1)
        temp_de = st.slider("DE temperature (°C)", 0.0, 120.0, 40.0, 1.0)
        temp_nde = st.slider("NDE temperature (°C)", 0.0, 120.0, 40.0, 1.0)

        st.subheader("Redundant sensor fusion (demo)")
        vib_alt = st.slider("Alt vibration sensor (mm/s)", 0.0, 10.0, 1.1, 0.1)

        # Availability risk score (0 best → 1 worst)
        # Explainable: weighted z-score against indicative thresholds
        vib_thr = 7.0
        temp_thr = 90.0
        disagreement = abs(vib - vib_alt)

        vib_risk = clamp(vib / vib_thr, 0, 1.5)
        temp_risk = clamp(max(temp_de, temp_nde) / temp_thr, 0, 1.5)
        sensor_risk = clamp(disagreement / 1.0, 0, 1.5)  # if sensors disagree strongly, data reliability drops

        # Combine
        availability_risk = 0.55 * vib_risk + 0.35 * temp_risk + 0.10 * sensor_risk
        availability_risk = clamp(availability_risk / 1.5, 0.0, 1.0)

        st.session_state["availability_risk_score"] = availability_risk

        if availability_risk < 0.25:
            st.success(f"Availability Risk: **Low** ({availability_risk:.2f}) — signals normal.")
            priority = "Routine"
        elif availability_risk < 0.50:
            st.warning(f"Availability Risk: **Medium** ({availability_risk:.2f}) — schedule inspection priority.")
            priority = "Elevated inspection priority"
        else:
            st.error(f"Availability Risk: **High** ({availability_risk:.2f}) — escalate to licensed operator review.")
            priority = "Escalate to operator review"

        st.info(f"Recommended action (prototype): **{priority}**")

        if show_assumptions:
            with st.expander("Assumptions (availability)"):
                st.markdown(
                    "- Thresholds are indicative placeholders.\n"
                    "- Risk score is an explainable proxy, not a diagnostic.\n"
                    "- Sensor disagreement is treated as data-quality risk.\n"
                    "- In real deployments, this layer would consume OEM-approved, non-safety-critical telemetry."
                )

    with right:
        # Trend visualization for vibration
        st.subheader("Trend view (simulated)")
        series = np.random.normal(1.2, 0.2, 50).tolist() + [vib]
        fig = px.line(series, labels={"index": "Time (hours)", "value": "Vibration (mm/s)"}, title="Vibration trend (simulated)")
        fig.add_hline(y=7.0, line_dash="dash", line_color="red", annotation_text="Indicative threshold")
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# TAB 5: Regulatory Navigator (RAG STUB, SAFE)
# -----------------------------
with tab5:
    st.header("Regulatory Navigator (RAG): Cited Answers + Uncertainty Flags")
    st.write("Prototype RAG module. In production, this would retrieve from a curated corpus of primary sources and policy notes.\n\n"
             "**Safeguard:** Decision support only; not legal advice.")

    # Minimal “document store” for demo (replace with real docs + embeddings later)
    # Structure: title, status (enacted/draft/guidance), snippet, source
    docs = [
        {
            "title": "Civil Liability for Nuclear Damage (CLND) Act, 2010 — Overview",
            "status": "Enacted law",
            "snippet": "Defines operator liability, claims, and recourse provisions; establishes liability framework for nuclear incidents.",
            "source": "Primary law text / official government publication"
        },
        {
            "title": "NITI Aayog: Role of SMRs in the Energy Transition (2023)",
            "status": "Guidance / policy report",
            "snippet": "Discusses SMR potential, deployment considerations, and system integration challenges.",
            "source": "NITI Aayog report"
        },
        {
            "title": "SHANTI Bill 2025 (Draft / Concept) — Placeholder",
            "status": "Draft / proposal (unverified)",
            "snippet": "Prototype placeholder. Replace with actual draft text if it exists; otherwise do not assert specifics as fact.",
            "source": "Draft / consultation doc (to be verified)"
        }
    ]
    df_docs = pd.DataFrame(docs)

    q = st.text_input("Enter a legal / policy question", "How does operator liability work under CLND Act 2010?")
    st.caption("Tip: Ask about licensing timelines, liability allocation, insurance, recourse, or private participation constraints.")

    # Naive retrieval: keyword match (demo-only)
    q_lower = q.lower()
    scores = []
    for i, row in df_docs.iterrows():
        text = f"{row['title']} {row['snippet']}".lower()
        score = sum(1 for w in q_lower.split() if w in text)
        scores.append(score)
    df_docs["match_score"] = scores
    top = df_docs.sort_values("match_score", ascending=False).head(2)

    col1, col2 = st.columns([1.2, 0.8])

    with col1:
        st.subheader("Aegis Regulatory Insight (prototype)")
        if top["match_score"].iloc[0] == 0:
            st.warning("No strong match found in the demo corpus. Add documents or refine the query.")
        else:
            for _, r in top.iterrows():
                st.markdown(f"**Source:** {r['title']}  \n**Status:** *{r['status']}*  \n**Relevance:** {int(r['match_score'])}")
                st.markdown(f"> {r['snippet']}")
                st.markdown(f"**Source note:** {r['source']}")
                st.markdown("---")

    with col2:
        st.subheader("Source Verification")
        st.progress(float(legal_coverage_score), text=f"Source Coverage Score: {int(legal_coverage_score*100)}%")
        st.caption("Higher = more complete primary sources retrieved and cross-checked (prototype input).")
        st.markdown("**Guardrails**")
        st.markdown(
            "- Separates **enacted law** vs **draft proposals** vs **guidance**.\n"
            "- Outputs cited snippets.\n"
            "- Flags low-coverage cases for human review."
        )

        if show_assumptions:
            with st.expander("What to build next (real RAG)"):
                st.markdown(
                    "- Replace demo corpus with curated PDFs/HTML of primary sources.\n"
                    "- Add embeddings + vector search.\n"
                    "- Add citation spans + “answerability” classifier.\n"
                    "- Add audit log for every query (investor-grade compliance)."
                )

# -----------------------------
# TAB 0: Executive Summary (builds on other tabs)
# -----------------------------
with tab0:
    st.header("Executive Decision Summary (Investor View)")

    demand_profile = st.session_state.get("demand_profile_mw", np.full(24, 10.0))
    licensing_samples = st.session_state.get("licensing_months_samples", np.random.normal(24, 6, 1000))
    be_samples = st.session_state.get("break_even_years_samples", np.random.normal(12, 2, 1000))
    availability_risk_score = st.session_state.get("availability_risk_score", 0.25)

    decision = build_decision(
        supply_mw=smr_rated_capacity,
        demand_profile=demand_profile,
        licensing_months_samples=licensing_samples,
        break_even_years_samples=be_samples,
        legal_coverage_score=legal_coverage_score,
        availability_risk_score=availability_risk_score,
    )

    # Decision tiles
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Aegis Verdict", decision.viability)
    c2.metric("Binding Constraint", decision.binding_constraint)
    c3.metric("Time-to-Power (P50–P90)", f"{fmt_range(decision.time_to_power_p50, decision.time_to_power_p90, ' mo')}")
    c4.metric("Break-even (P50–P90)", f"{fmt_range(decision.break_even_p50_years, decision.break_even_p90_years, ' yrs')}")

    # Confidence + recommendation
    if decision.confidence == "High":
        st.success(f"Confidence: **{decision.confidence}**")
    elif decision.confidence == "Medium":
        st.warning(f"Confidence: **{decision.confidence}**")
    else:
        st.error(f"Confidence: **{decision.confidence}**")

    st.info(f"**Recommendation:** {decision.recommendation}")

    # Explain “why” (investor-friendly)
    st.subheader("Why Aegis reached this verdict (explainable drivers)")
    peak = float(np.max(demand_profile))
    avg = float(np.mean(demand_profile))
    supply_ratio_peak = smr_rated_capacity / max(peak, 1e-6)
    supply_ratio_avg = smr_rated_capacity / max(avg, 1e-6)

    d1, d2, d3 = st.columns(3)
    d1.metric("Supply ratio (peak)", f"{supply_ratio_peak:.2f}×")
    d2.metric("Supply ratio (avg)", f"{supply_ratio_avg:.2f}×")
    d3.metric("Regulatory coverage", f"{int(legal_coverage_score*100)}%")

    st.caption("Interpretation: SMR is treated as fixed baseload; routing flexibility is downstream (battery/thermal/cooling).")

    # Quick comparison baseline (Grid + Battery) – very simple, but powerful for pitch
    st.subheader("Baseline Comparator (Grid + Storage) — prototype lens")
    grid_time_years = st.slider("Grid upgrade timeline (years)", 1.0, 10.0, 6.0, 0.5)
    storage_cost_proxy = st.slider("Storage cost proxy (₹ crore) — for reliability", 0.0, 5000.0, 800.0, 50.0)

    sm_time_years = decision.time_to_power_p50 / 12.0
    sm_time_p90_years = decision.time_to_power_p90 / 12.0

    comp = pd.DataFrame({
        "Option": ["SMR co-location (P50)", "SMR co-location (P90)", "Grid upgrade"],
        "Time-to-power (years)": [sm_time_years, sm_time_p90_years, grid_time_years],
        "Reliability add-on cost proxy (₹ crore)": [0.0, 0.0, storage_cost_proxy]
    })
    st.dataframe(comp, use_container_width=True)

    st.markdown("---")
    st.markdown(
        "**Boundary & disclaimer:** Aegis is a **decision-support prototype**. Inputs are simulated for demonstration. "
        "It provides planning insights and uncertainty framing; it **does not control or operate nuclear assets**."
    )

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown(
    "**Aegis Cognitive Twin (Prototype)** — AI-native decision support for Nuclear-as-a-Service (NaaS) + AI infrastructure.\n\n"
    "**Scope boundary:** Advises planners/investors/operators via simulation & cited policy retrieval. "
    "**Not** a reactor control or licensed operations system."
)