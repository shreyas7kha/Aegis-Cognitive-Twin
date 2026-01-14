import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Aegis Cognitive Twin", layout="wide")


# -----------------------------
# HELPERS
# -----------------------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def percentile(x, p):
    return float(np.percentile(np.asarray(x), p))


def fmt_range(p50, p90, unit=""):
    if unit:
        return f"{p50:.0f}{unit}–{p90:.0f}{unit}"
    return f"{p50:.0f}–{p90:.0f}"


def annuity_factor(r, n):
    if r <= 0:
        return float(n)
    return (r * (1 + r) ** n) / ((1 + r) ** n - 1)


def badge(text, kind="info"):
    if kind == "success":
        st.success(text)
    elif kind == "warning":
        st.warning(text)
    elif kind == "error":
        st.error(text)
    else:
        st.info(text)


def build_decision(
    supply_mw_effective: float,
    demand_profile_mw: np.ndarray,
    ttp_months_samples: np.ndarray,
    break_even_years_samples: np.ndarray,
    legal_coverage_score: float,
    availability_risk_score: float,
):
    """
    Investor-style decision logic: simple, explainable.
    This is decision support (prototype). No operational control.
    """
    peak = float(np.max(demand_profile_mw))
    avg = float(np.mean(demand_profile_mw))

    supply_ratio_peak = supply_mw_effective / max(peak, 1e-6)
    supply_ratio_avg = supply_mw_effective / max(avg, 1e-6)

    ttp_p50 = percentile(ttp_months_samples, 50)
    ttp_p90 = percentile(ttp_months_samples, 90)
    be_p50 = percentile(break_even_years_samples, 50)
    be_p90 = percentile(break_even_years_samples, 90)

    # Transparent thresholds tuned for demo narrative (adjust later with stakeholder input)
    supply_flag = (supply_ratio_peak >= 1.0) or (supply_ratio_avg >= 0.90)
    ttp_flag = (ttp_p90 <= 42)          # months
    econ_flag = (be_p90 <= 18)          # years
    reg_flag = (legal_coverage_score >= 0.70)
    avail_flag = (availability_risk_score <= 0.35)  # lower is better

    flags = {
        "Supply adequacy": supply_flag,
        "Time-to-power": ttp_flag,
        "Economics": econ_flag,
        "Regulatory coverage": reg_flag,
        "Availability risk": avail_flag
    }
    failed = [k for k, v in flags.items() if not v]

    # Pick binding constraint by severity (simple, but explainable)
    sev = {
        "Supply adequacy": (0 if supply_flag else (1.0 - supply_ratio_avg)),
        "Time-to-power": (0 if ttp_flag else (ttp_p90 - 42) / 24),
        "Economics": (0 if econ_flag else (be_p90 - 18) / 10),
        "Regulatory coverage": (0 if reg_flag else (0.7 - legal_coverage_score)),
        "Availability risk": (0 if avail_flag else (availability_risk_score - 0.35))
    }
    binding_constraint = max(sev, key=sev.get)

    # Confidence (prototype): mostly source coverage + data quality (availability signal)
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

    return {
        "viability": viability,
        "binding_constraint": binding_constraint,
        "ttp_p50": ttp_p50,
        "ttp_p90": ttp_p90,
        "be_p50": be_p50,
        "be_p90": be_p90,
        "confidence": confidence,
        "recommendation": recommendation,
        "supply_ratio_peak": supply_ratio_peak,
        "supply_ratio_avg": supply_ratio_avg,
        "peak_demand": peak,
        "avg_demand": avg,
    }


# -----------------------------
# SIDEBAR (global controls, but minimal)
# -----------------------------
st.sidebar.title("Aegis Control Panel")
st.sidebar.markdown("---")
st.sidebar.info(
    "Disclaimer: **Aegis advises; it does not control nuclear assets.**"
)

# Keep globals only for things that truly are “system-level knobs”
smr_unit_rated_mw = st.sidebar.number_input("SMR Unit Rated Capacity (MW)", 15, 300, 77, step=1)
discount_rate = st.sidebar.slider("Discount Rate (%)", 3.0, 18.0, 10.0, 0.5) / 100.0
downtime_cost_per_hr_crore = st.sidebar.number_input("Downtime Cost (₹ crore / hour)", 0.1, 50.0, 5.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("Regulatory Coverage")
legal_coverage_score = st.sidebar.slider(
    "Source Coverage Score (0–1)", 0.0, 1.0, 0.85, 0.01,
    help="Proxy for completeness of retrieved primary sources in the RAG module."
)

st.sidebar.markdown("---")
show_assumptions = st.sidebar.checkbox("Show assumptions panels", value=True)

# -----------------------------
# HEADER
# -----------------------------
st.title("🛡️ Aegis Cognitive Twin: Decision Support Engine")
st.subheader("AI-native platform to evaluate SMR co-location for AI data centres under uncertainty")


# -----------------------------
# TABS
# -----------------------------
tab_exec, tab_workflow, tab_facility, tab_econ, tab_risk, tab_assets, tab_reg = st.tabs([
    "✅ Executive Summary",
    "📈 Demand Workflow",
    "🏭 Supply Workflow",
    "💰 Economics",
    "⏱️ Risk Analysis",
    "🧭 Predictive Intelligence",
    "⚖️ Regulatory Navigator (RAG)"
])


# -----------------------------
# TAB: Workload & Workflow
# -----------------------------
with tab_workflow:
    st.header("Demand Intelligence")
    left, right = st.columns([1, 2])

    with left:
        st.write("### Scenario Presets")
        scenario = st.selectbox(
            "Select workload regime",
            [
                "Inference-dominant steady state",
                "Fine-tuning sprint (bursty)",
                "Pre-training week (high utilization)",
                "Mixed enterprise workload (spiky)"
            ],
            key="wf_scenario"
        )

        st.write("### Compute & Facility Parameters")
        gpu_count = st.slider("AI Cluster Size (H100 GPUs)", 1000, 200000, 10000, step=1000, key="wf_gpu")
        pue_target = st.slider("Target PUE", 1.0, 2.0, 1.2, 0.05, key="wf_pue")
        cooling_fraction = st.slider("Cooling share of total facility load (%)", 10, 60, 30, 1, key="wf_cool")

        # Regime parameters (local)
        if scenario == "Inference-dominant steady state":
            base_util = st.slider("Average utilization (%)", 10, 60, 25, 1, key="wf_util")
            peakiness = 0.15
            burst_prob = 0.08
        elif scenario == "Fine-tuning sprint (bursty)":
            base_util = st.slider("Average utilization (%)", 30, 85, 60, 1, key="wf_util")
            peakiness = 0.35
            burst_prob = 0.22
        elif scenario == "Pre-training week (high utilization)":
            base_util = st.slider("Average utilization (%)", 60, 95, 85, 1, key="wf_util")
            peakiness = 0.20
            burst_prob = 0.12
        else:
            base_util = st.slider("Average utilization (%)", 20, 80, 50, 1, key="wf_util")
            peakiness = 0.45
            burst_prob = 0.25

        # Simple power model (local)
        gpu_kw_proxy = st.slider("GPU power proxy (kW per GPU)", 0.4, 1.2, 0.7, 0.05, key="wf_gkW")
        it_power_mw_at_100 = (gpu_count * gpu_kw_proxy) / 1000.0
        avg_demand_mw = it_power_mw_at_100 * (base_util / 100.0) * pue_target

        # Fleet sizing guidance (talks to facility tab later)
        req_units_avg = int(np.ceil(avg_demand_mw / max(float(smr_unit_rated_mw), 1e-6)))
        st.metric("Average predicted facility demand", f"{avg_demand_mw:.2f} MW")
        st.info(f"Indicative SMR fleet (avg basis): **{req_units_avg} unit(s)** @ {smr_unit_rated_mw} MW each")

        if show_assumptions:
            with st.expander("Model assumptions"):
                st.markdown(
                    "- GPU power is a **proxy** for demo. Real deployments replace this with measured IT load.\n"
                    "- Facility demand = IT power × utilization × PUE.\n"
                    "- Volatility is simulated via bursts + small noise.\n"
                    "- Outputs are decision-support only; validate with site engineering studies."
                )

    # Build 24-hour demand profile (local)
    rng = np.random.default_rng(7)
    times = pd.date_range("2026-01-01", periods=24, freq="h")
    demand = []
    for _ in range(24):
        burst = (rng.random() < burst_prob)
        mult = 1.0 + (peakiness * (0.8 + 0.4 * rng.random()) if burst else 0.0)
        noise = rng.normal(0, 0.03)
        d = avg_demand_mw * (mult + noise)
        demand.append(max(0.0, d))
    demand = np.asarray(demand, dtype=float)

    # Store minimal outputs for other tabs
    st.session_state["demand_profile_mw"] = demand
    st.session_state["avg_demand_mw"] = float(np.mean(demand))
    st.session_state["peak_demand_mw"] = float(np.max(demand))
    st.session_state["pue_target"] = float(pue_target)
    st.session_state["scenario"] = scenario
    st.session_state["req_units_avg"] = int(req_units_avg)

    with right:
        supply_line = np.full(24, float(smr_unit_rated_mw))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=demand, name="Facility demand (MW)", line=dict(width=3)))
        fig.add_trace(go.Scatter(x=times, y=supply_line, name="1× SMR baseload (MW)", line=dict(dash="dash", width=3)))
        fig.update_layout(
            title="24-hour demand vs 1× SMR unit baseload (for intuition)",
            yaxis_title="Power (MW)",
            xaxis_title="Time"
        )
        st.plotly_chart(fig)

        peak = float(np.max(demand))
        st.caption(
            f"Peak demand (simulated): **{peak:.2f} MW** | "
            f"Supply ratio (peak) for 1 unit: **{float(smr_unit_rated_mw)/max(peak,1e-6):.2f}×**"
        )


# -----------------------------
# TAB: Select Facility
# -----------------------------
with tab_facility:
    st.header("Supply Intelligence")
    demand_profile = st.session_state.get("demand_profile_mw")
    if demand_profile is None:
        st.warning("Go to 'Workload & Workflow' tab first.")
    else:
        left, right = st.columns([1, 2])

        with left:
            st.write("### Facility Sizing Inputs")
            # Fleet sizing talks to demand
            sizing_basis = st.selectbox("Fleet sizing basis", ["Average demand", "Peak demand"], key="sf_basis")
            if sizing_basis == "Average demand":
                required_units = int(np.ceil(st.session_state["avg_demand_mw"] / max(float(smr_unit_rated_mw), 1e-6)))
                basis_value = st.session_state["avg_demand_mw"]
            else:
                required_units = int(np.ceil(st.session_state["peak_demand_mw"] / max(float(smr_unit_rated_mw), 1e-6)))
                basis_value = st.session_state["peak_demand_mw"]

            fleet_units = st.slider("Number of SMR units (override)", 1, 10, max(1, required_units), 1, key="sf_units")
            cf = st.slider("Availability / capacity factor", 0.70, 0.98, 0.90, 0.01, key="sf_cf")

            st.caption(f"Suggested units by {sizing_basis.lower()} = **{required_units}** (basis demand: {basis_value:.2f} MW)")

            st.write("### Storage Capability")
            battery_power_mw = st.slider("Battery power capacity (MW)", 0.0, 200.0, 20.0, 1.0, key="sf_bp")
            battery_hours = st.slider("Battery duration (hours @ rated power)", 0.5, 12.0, 2.0, 0.5, key="sf_bh")
            thermal_absorb_mw = st.slider("Thermal storage absorption (MW equiv.)", 0.0, 200.0, 15.0, 1.0, key="sf_th")

            soc_pct = st.slider("Battery state of charge (%)", 0, 100, 50, 1, key="sf_soc")
            soc_frac = soc_pct / 100.0

            if show_assumptions:
                with st.expander("Assumptions (facility feasibility)"):
                    st.markdown(
                        "- SMR treated as **flat baseload** at (units × rated × CF).\n"
                        "- Battery modeled with power cap and energy cap (power × hours).\n"
                        "- Thermal storage shown as a sink (absorb surplus) proxy.\n"
                        "- This is a planning feasibility lens, not a microgrid power-flow model."
                    )

        # Local computation: 24h dispatch feasibility
        supply_mw = float(fleet_units) * float(smr_unit_rated_mw) * float(cf)
        demand = np.asarray(demand_profile, dtype=float)
        batt_energy_mwh = float(battery_power_mw) * float(battery_hours)
        soc_mwh = soc_frac * batt_energy_mwh

        batt_charge = []
        batt_discharge = []
        thermal_to = []
        curtailed = []
        unserved = []
        soc_series = []

        for t in range(24):
            net = supply_mw - demand[t]
            if net >= 0:
                # Charge battery then thermal then curtail
                ch = min(net, float(battery_power_mw), (batt_energy_mwh - soc_mwh)) if batt_energy_mwh > 0 else 0.0
                soc_mwh += ch
                rem = net - ch
                th = min(rem, float(thermal_absorb_mw))
                rem2 = rem - th
                batt_charge.append(ch)
                batt_discharge.append(0.0)
                thermal_to.append(th)
                curtailed.append(max(0.0, rem2))
                unserved.append(0.0)
            else:
                deficit = abs(net)
                dis = min(deficit, float(battery_power_mw), soc_mwh) if batt_energy_mwh > 0 else 0.0
                soc_mwh -= dis
                rem = deficit - dis
                batt_charge.append(0.0)
                batt_discharge.append(dis)
                thermal_to.append(0.0)
                curtailed.append(0.0)
                unserved.append(max(0.0, rem))

            soc_series.append((soc_mwh / batt_energy_mwh) * 100.0 if batt_energy_mwh > 0 else 0.0)

        facility_df = pd.DataFrame({
            "hour": np.arange(24),
            "Demand_MW": demand,
            "Supply_MW": np.full(24, supply_mw),
            "Battery_charge_MW": batt_charge,
            "Battery_discharge_MW": batt_discharge,
            "Thermal_absorb_MW": thermal_to,
            "Curtailed_MW": curtailed,
            "Unserved_MW": unserved,
            "SOC_pct": soc_series
        })

        energy_total = float(np.sum(demand))
        unserved_total = float(np.sum(facility_df["Unserved_MW"]))
        reliability = 1.0 - (unserved_total / max(energy_total, 1e-9))
        reliability = clamp(reliability, 0.0, 1.0)

        # Store minimal outputs
        st.session_state["facility_df"] = facility_df
        st.session_state["supply_mw_effective"] = float(supply_mw)
        st.session_state["fleet_units"] = int(fleet_units)
        st.session_state["reliability"] = float(reliability)

        with right:
            st.write("### Feasibility View")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=facility_df["hour"], y=facility_df["Demand_MW"], name="Demand", line=dict(width=3)))
            fig.add_trace(go.Scatter(x=facility_df["hour"], y=facility_df["Supply_MW"], name="Supply (baseload)", line=dict(dash="dash", width=3)))
            fig.add_trace(go.Bar(x=facility_df["hour"], y=facility_df["Battery_discharge_MW"], name="Battery discharge", opacity=0.6))
            fig.add_trace(go.Bar(x=facility_df["hour"], y=-facility_df["Battery_charge_MW"], name="Battery charge (neg)", opacity=0.6))
            fig.add_trace(go.Bar(x=facility_df["hour"], y=facility_df["Thermal_absorb_MW"], name="Thermal absorb", opacity=0.35))
            fig.add_trace(go.Bar(x=facility_df["hour"], y=facility_df["Unserved_MW"], name="Unserved", opacity=0.85))
            fig.update_layout(barmode="relative", xaxis_title="Hour", yaxis_title="MW")
            st.plotly_chart(fig)

            fig_soc = px.line(facility_df, x="hour", y="SOC_pct", title="Battery state-of-charge")
            st.plotly_chart(fig_soc)

            c1, c2, c3 = st.columns(3)
            c1.metric("Effective supply (MW)", f"{supply_mw:.2f}")
            c2.metric("Reliability (energy-based)", f"{reliability*100:.3f}%")
            c3.metric("Unserved (MWh/day)", f"{unserved_total:.2f}")


# -----------------------------
# TAB: Economics (with clear implied value decomposition)
# -----------------------------
with tab_econ:
    st.header("Economics")
    facility_df = st.session_state.get("facility_df")
    if facility_df is None:
        st.warning("Go to 'Supply Workflow' tab first.")
    else:
        left, right = st.columns([1, 2])

        with left:
            st.write("### Financial Inputs")
            capex_per_mw_crore = st.number_input("All-in CAPEX (₹ crore / MW)", 1.0, 100.0, 10.0, 0.5, key="ec_capex")
            fixed_capex_crore = st.number_input("Fixed CAPEX (₹ crore)", 0.0, 50000.0, 100.0, 50.0, key="ec_fixcapex")
            opex_fixed_crore_per_year = st.number_input("Fixed OPEX (₹ crore / year)", 0.0, 20000.0, 100.0, 10.0, key="ec_opexfix")

            opex_per_kwh = st.number_input("Variable OPEX (₹ / kWh)", 0.1, 10.0, 1.0, 0.1, key="ec_opexkwh")

            project_life_years = st.slider("Project life (years)", 10, 40, 25, key="ec_life")
            wacc = st.slider("Discount rate / WACC (%)", 3.0, 18.0, 10.0, 0.5, key="ec_wacc") / 100.0

            st.write("### Value inputs")
            grid_tariff = st.number_input("Grid tariff (₹ / kWh)", 1.0, 30.0, 8.0, 0.5, key="ec_grid")
            backup_tariff = st.number_input("Backup / diesel tariff (₹ / kWh)", 5.0, 60.0, 18.0, 0.5, key="ec_backup")
            outage_base_hours = st.slider("Outage hours/year (baseline)", 0.0, 200.0, 40.0, 1.0, key="ec_out0")
            outage_with_aegis_hours = st.slider("Outage hours/year (with SMR+storage)", 0.0, 100.0, 6.0, 1.0, key="ec_out1")

            if show_assumptions:
                with st.expander("Interpretation"):
                    st.markdown(
                        "**Implied value** is not a market price claim. It's a *business value proxy*:\n\n"
                        "1) **Energy value**: kWh served × grid tariff (what you'd otherwise pay).\n"
                        "2) **Backup avoided premium**: kWh that would have been supplied by backup × (backup − grid).\n"
                        "3) **Downtime avoided value**: avoided outage hours × downtime cost.\n\n"
                        "This turns reliability into ₹ impact — the part investors care about."
                    )

        # Local computations
        fleet_units = int(st.session_state.get("fleet_units", 1))
        supply_mw_effective = float(st.session_state.get("supply_mw_effective", 0.0))
        avg_demand_mw = float(st.session_state.get("avg_demand_mw", float(facility_df["Demand_MW"].mean())))

        # Approximate nameplate MW for capex (use rated units, not CF-adjusted)
        nameplate_mw = fleet_units * float(smr_unit_rated_mw)

        total_capex_crore = (nameplate_mw * capex_per_mw_crore) + fixed_capex_crore
        annualized_capex_crore = total_capex_crore * annuity_factor(wacc, int(project_life_years))

        annual_mwh_served = float(facility_df["Demand_MW"].sum()) * 365.0
        annual_kwh_served = annual_mwh_served * 1000.0

        annual_opex_crore = opex_fixed_crore_per_year + (opex_per_kwh * annual_kwh_served) / 1e7

        # LCOE (₹/kWh): (annualized capex + annual opex) / annual kWh
        lcoe_rs_per_kwh = ((annualized_capex_crore + annual_opex_crore) * 1e7) / max(annual_kwh_served, 1e-9)

        # Implied value decomposition
        energy_value_rs = annual_kwh_served * grid_tariff
        backup_kwh_avoided = max(0.0, (outage_base_hours - outage_with_aegis_hours) * avg_demand_mw * 1000.0)
        backup_value_rs = backup_kwh_avoided * max(0.0, (backup_tariff - grid_tariff))
        downtime_value_rs = max(0.0, (outage_base_hours - outage_with_aegis_hours)) * downtime_cost_per_hr_crore * 1e7

        implied_value_rs = energy_value_rs + backup_value_rs + downtime_value_rs
        implied_rs_per_kwh = implied_value_rs / max(annual_kwh_served, 1e-9)

        # Store minimal outputs for risk + summary
        st.session_state["lcoe_rs_per_kwh"] = float(lcoe_rs_per_kwh)
        st.session_state["implied_rs_per_kwh"] = float(implied_rs_per_kwh)

        with right:
            st.write("### Outputs")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total CAPEX (₹ crore)", f"{total_capex_crore:,.0f}")
            c2.metric("Annualized CAPEX (₹ crore/yr)", f"{annualized_capex_crore:,.0f}")
            c3.metric("Annual OPEX (₹ crore/yr)", f"{annual_opex_crore:,.0f}")
            c4.metric("LCOE (₹/kWh)", f"{lcoe_rs_per_kwh:.2f}")

            d1, d2, d3 = st.columns(3)
            d1.metric("Implied value (₹/kWh)", f"{implied_rs_per_kwh:.2f}")
            d2.metric("Value − LCOE (₹/kWh)", f"{(implied_rs_per_kwh - lcoe_rs_per_kwh):.2f}")
            d3.metric("Fleet (units)", f"{fleet_units}")

            parts = pd.DataFrame({
                "Component": ["Energy value (grid)", "Backup avoided premium", "Downtime avoided value"],
                "₹/year": [energy_value_rs, backup_value_rs, downtime_value_rs]
            })
            st.plotly_chart(px.bar(parts, x="Component", y="₹/year", title="Implied value decomposition"))

            with st.expander("Explain in 20 seconds"):
                st.markdown(
                    f"- Your cost of electricity (LCOE proxy) is **₹{lcoe_rs_per_kwh:.2f}/kWh**.\n"
                    f"- The business value proxy (energy + avoided backup + avoided downtime) is **₹{implied_rs_per_kwh:.2f}/kWh**.\n"
                    f"- The spread (**₹{(implied_rs_per_kwh - lcoe_rs_per_kwh):.2f}/kWh**) drives whether this is worth piloting."
                )


# -----------------------------
# TAB: Risk (Time-to-power + break-even cloud) + interpretation
# -----------------------------
with tab_risk:
    st.header("Risk Analysis")
    st.subheader("Time-to-Power + Break-even under Uncertainty")

    demand_profile = st.session_state.get("demand_profile_mw", np.full(24, 10.0))
    avg_demand_mw = float(st.session_state.get("avg_demand_mw", float(np.mean(demand_profile))))

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Simulation Parameters")
        sim_count = st.slider("Simulation runs", 200, 5000, 1500, 100, key="rk_n")

        licensing_mean = st.slider("Licensing mean (months)", 6, 60, 24, 1, key="rk_lm")
        licensing_std = st.slider("Licensing std dev (months)", 1, 24, 6, 1, key="rk_ls")

        construction_mean = st.slider("Construction mean (months)", 6, 60, 18, 1, key="rk_cm")
        construction_std = st.slider("Construction std dev (months)", 1, 24, 5, 1, key="rk_cs")

        delay_prob = st.slider("Delay probability", 0.0, 1.0, 0.25, 0.05, key="rk_dp")
        delay_months = st.slider("Delay months if hit", 0, 24, 6, 1, key="rk_dm")

        capex_unc_sd = st.slider("CAPEX uncertainty (sd %)", 0.0, 40.0, 15.0, 1.0, key="rk_csd")
        margin_unc_sd = st.slider("Value uncertainty (sd %)", 0.0, 60.0, 20.0, 1.0, key="rk_vsd")

        if show_assumptions:
            with st.expander("How to interpret the cloud"):
                st.markdown(
                    "**Each dot** = one possible future scenario.\n\n"
                    "- **X-axis (time-to-power)**: how long until you can actually deliver power.\n"
                    "- **Y-axis (break-even proxy)**: how long until benefits pay back the investment.\n\n"
                    "So the cloud tells you: **how often do we end up late and expensive?**\n\n"
                    "**Inputs that move the cloud:** licensing+construction uncertainty, delay probability, CAPEX uncertainty, and value uncertainty."
                )

    # Local simulation
    rng = np.random.default_rng(42)
    lic = np.clip(rng.normal(licensing_mean, licensing_std, sim_count), 3, 84)
    con = np.clip(rng.normal(construction_mean, construction_std, sim_count), 3, 84)
    delay = (rng.random(sim_count) < delay_prob).astype(float) * float(delay_months)

    time_to_power_months = lic + con + delay
    time_to_power_months = np.clip(time_to_power_months, 6, 120)

    # Use economics results if available; otherwise fallback to placeholders
    lcoe = float(st.session_state.get("lcoe_rs_per_kwh", 9.0))
    implied = float(st.session_state.get("implied_rs_per_kwh", 10.0))

    capex_mult = np.clip(rng.normal(1.0, capex_unc_sd / 100.0, sim_count), 0.6, 1.8)
    value_mult = np.clip(rng.normal(1.0, margin_unc_sd / 100.0, sim_count), 0.4, 2.2)

    value_minus_cost = (implied * value_mult) - (lcoe * capex_mult)

    # Break-even proxy: (capex pressure + delay penalty) / margin pressure (very simplified but coherent)
    delay_years = time_to_power_months / 12.0
    delay_penalty = (1 + discount_rate) ** delay_years

    # “Annual benefit” proxy (scaled) — we don't pretend this is project finance
    benefit_proxy = np.maximum(0.1, 2.0 * value_minus_cost)  # arbitrary scaling for demo stability
    break_even_years = np.clip((10.0 * capex_mult * delay_penalty) / benefit_proxy, 1.0, 60.0)

    risk_df = pd.DataFrame({
        "time_to_power_months": time_to_power_months,
        "break_even_years": break_even_years,
        "value_minus_cost": value_minus_cost
    })

    # Store for summary/decision
    st.session_state["ttp_months_samples"] = time_to_power_months
    st.session_state["break_even_years_samples"] = break_even_years

    with right:
        st.subheader("Outputs")
        p50_ttp = percentile(time_to_power_months, 50)
        p90_ttp = percentile(time_to_power_months, 90)
        p50_be = percentile(break_even_years, 50)
        p90_be = percentile(break_even_years, 90)

        c1, c2 = st.columns(2)
        c1.metric("Time-to-power (P50–P90)", fmt_range(p50_ttp, p90_ttp, " mo"))
        c2.metric("Break-even (P50–P90)", fmt_range(p50_be, p90_be, " yrs"))

        st.plotly_chart(
            px.histogram(risk_df, x="time_to_power_months", nbins=35, histnorm="probability",
                         title="Time-to-power distribution")
        )

        st.plotly_chart(
            px.scatter(
                risk_df, x="time_to_power_months", y="break_even_years",
                color="value_minus_cost", opacity=0.6,
                title="Delay vs Break-even",
                labels={"time_to_power_months": "Months to operational", "break_even_years": "Years to break-even"}
            )
        )


# -----------------------------
# TAB: Asset Availability Intelligence (old UI feel + new signals)
# -----------------------------
with tab_assets:
    st.header("Asset Health Intelligence")
    st.warning("Disclaimer: Aegis **does not** diagnose faults or prescribe maintenance. It flags indicative risks for operator review.")

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Equipment Diagnostics")
        vib = st.slider("Pump vibration (mm/s)", 0.0, 10.0, 1.2, 0.1, key="as_v1")
        temp_de = st.slider("Pump DE temperature (°C)", 0.0, 120.0, 40.0, 1.0, key="as_tde")
        temp_nde = st.slider("Pump NDE temperature (°C)", 0.0, 120.0, 40.0, 1.0, key="as_tnde")
        pressure_delta = st.slider("Pressure delta (bar)", 0.0, 10.0, 1.6, 0.1, key="as_pd")
        pump_life = st.slider("Total Useful Life (years)", 0.0, 15.0, 2.0, 0.1, key="as_pumplife")
        cadence = st.selectbox("Cadence", ["1m", "10m", "30m" "1h"], index=1, key="as_cad")

        # New-ish signals (still safe): priority events based on thresholds
        events = []
        if vib > 4.5:
            events.append("High vibration event")
        if max(temp_de, temp_nde) > 70:
            events.append("High temperature event")
        if pressure_delta > 4.0:
            events.append("Pressure delta anomaly")

        if len(events) == 0:
            events = ["No priority events detected"]

        st.subheader("Priority events")
        for e in events:
            st.write(f"• {e}")

    # Explainable risk score (like old version, but improved)
    vib_thr = 7.0
    temp_thr = 90.0

    vib_risk = clamp(vib / vib_thr, 0, 1.5)
    temp_risk = clamp(max(temp_de, temp_nde) / temp_thr, 0, 1.5)
    pd_risk = clamp(pressure_delta / 6.0, 0, 1.5)

    availability_risk = 0.5 * vib_risk + 0.35 * temp_risk + 0.15 * pd_risk

    cadence_factor = {"1m": 1.00, "10m": 0.93, "30m":0.89, "1h": 0.85}[cadence]
    availability_risk = clamp((availability_risk / 1.5) * cadence_factor, 0.0, 1.0)

    # Add M-score + RUL (visible)
    m_score = clamp(availability_risk * 100.0, 0.0, 100.0)
    rul_days = int(clamp(pump_life * 365 * (1.0 - (m_score / 100.0)) ** 1.6, 10, 365))

    # Store minimal for decision confidence
    st.session_state["availability_risk_score"] = float(availability_risk)
    st.session_state["m_score"] = float(m_score)
    st.session_state["rul_days"] = int(rul_days)

    with right:
        st.subheader("Availability Dashboard")
        if availability_risk < 0.25:
            badge(f"Availability Risk: **Low** ({availability_risk:.2f}) — signals normal.", "success")
            priority = "Routine"
        elif availability_risk < 0.50:
            badge(f"Availability Risk: **Medium** ({availability_risk:.2f}) — schedule inspection priority.", "warning")
            priority = "Elevated inspection priority"
        else:
            badge(f"Availability Risk: **High** ({availability_risk:.2f}) — escalate to licensed operator review.", "error")
            priority = "Escalate to operator review"

        st.info(f"Recommended action: **{priority}**")

        c1, c2, c3 = st.columns(3)
        c1.metric("M-score", f"{m_score:.1f}")
        c2.metric("RUL (days)", f"{rul_days}")
        c3.metric("Data cadence", cadence)

        # Trend (simulated) — like old UI
        series = np.random.default_rng().normal(1.2, 0.2, 50).tolist() + [vib]
        fig = px.line(series, labels={"index": "Time (hours)", "value": "Vibration (mm/s)"}, title="Vibration trend")
        fig.add_hline(y=vib_thr, line_dash="dash", line_color="red", annotation_text="Indicative threshold")
        st.plotly_chart(fig)

        if show_assumptions:
            with st.expander("Assumptions (availability)"):
                st.markdown(
                    "- Thresholds are indicative placeholders.\n"
                    "- Risk score is an explainable proxy, not a diagnostic.\n"
                    "- Real deployments should consume OEM-approved, non-safety-critical telemetry."
                )


# -----------------------------
# TAB: Regulatory Navigator (RAG stub, better looking)
# -----------------------------
with tab_reg:
    st.header("Regulatory Navigator (RAG)")
    st.subheader("Cited Answers + Uncertainty Flags")
    st.warning(
        "**Safeguard:** Decision support only; not legal advice."
    )

    docs = [
        {
            "title": "Civil Liability for Nuclear Damage (CLND) Act, 2010 — Overview",
            "status": "Enacted law",
            "snippet": "Defines operator liability, claims, and recourse provisions; establishes liability framework for nuclear incidents.",
            "source": "Primary law text / official government publication"
        },
        {
            "title": "NITI Aayog: Role of SMRs in the Energy Transition (Policy report)",
            "status": "Guidance / policy report",
            "snippet": "Discusses SMR potential, deployment considerations, and system integration challenges.",
            "source": "NITI Aayog report"
        },
        {
            "title": "Draft / consultation paper — verify before relying",
            "status": "Draft / proposal",
            "snippet": "Prototype placeholder. Replace with actual draft text if it exists; otherwise do not assert specifics as fact.",
            "source": "Consultation / draft doc"
        }
    ]
    df_docs = pd.DataFrame(docs)

    q = st.text_input("Enter a legal / policy question", "How does operator liability work under CLND Act 2010?", key="rg_q")
    st.caption("Tip: Ask about licensing timelines, liability allocation, insurance, recourse, or private participation constraints.")

    q_lower = q.lower().strip()
    scores = []
    for _, row in df_docs.iterrows():
        text = f"{row['title']} {row['snippet']}".lower()
        score = sum(1 for w in q_lower.split() if w in text)
        scores.append(score)
    df_docs["match_score"] = scores
    top = df_docs.sort_values("match_score", ascending=False).head(2)

    col1, col2 = st.columns([1.2, 0.8])

    with col1:
        st.subheader("Aegis Regulatory Insight")
        if int(top["match_score"].iloc[0]) == 0:
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
        st.caption("Higher = more complete primary sources retrieved and cross-checked.")

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
                    "- Add citation spans + answerability classifier.\n"
                    "- Add audit log for every query (investor-grade compliance)."
                )


# -----------------------------
# TAB: Executive Summary (best of old UI + new outputs)
# -----------------------------
with tab_exec:
    st.header("Executive Decision Summary")

    demand_profile = st.session_state.get("demand_profile_mw", np.full(24, 10.0))
    supply_mw_effective = float(st.session_state.get("supply_mw_effective", float(smr_unit_rated_mw)))
    availability_risk_score = float(st.session_state.get("availability_risk_score", 0.25))

    ttp_samples = st.session_state.get("ttp_months_samples")
    be_samples = st.session_state.get("break_even_years_samples")

    # Provide safe fallbacks if risk tab not visited
    if ttp_samples is None:
        rng = np.random.default_rng(11)
        ttp_samples = np.clip(rng.normal(36, 8, 1200), 6, 120)
    if be_samples is None:
        rng = np.random.default_rng(12)
        be_samples = np.clip(rng.normal(14, 4, 1200), 1, 60)

    decision = build_decision(
        supply_mw_effective=supply_mw_effective,
        demand_profile_mw=demand_profile,
        ttp_months_samples=np.asarray(ttp_samples),
        break_even_years_samples=np.asarray(be_samples),
        legal_coverage_score=float(legal_coverage_score),
        availability_risk_score=float(availability_risk_score),
    )

    # --- Top tiles (old style)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Aegis Verdict", decision["viability"])
    c2.metric("Binding Constraint", decision["binding_constraint"])
    c3.metric("Time-to-Power (P50–P90)", fmt_range(decision["ttp_p50"], decision["ttp_p90"], " mo"))
    c4.metric("Break-even (P50–P90)", fmt_range(decision["be_p50"], decision["be_p90"], " yrs"))

    if decision["confidence"] == "High":
        st.success(f"Confidence: **{decision['confidence']}**")
    elif decision["confidence"] == "Medium":
        st.warning(f"Confidence: **{decision['confidence']}**")
    else:
        st.error(f"Confidence: **{decision['confidence']}**")

    st.info(f"**Recommendation:** {decision['recommendation']}")

    st.markdown("---")

    # --- Nuclear highlights / key parameters (the “interesting” part you asked for)
    st.subheader("Key Parameters")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Supply ratio (avg)", f"{decision['supply_ratio_avg']:.2f}×")
    k2.metric("Supply ratio (peak)", f"{decision['supply_ratio_peak']:.2f}×")
    k3.metric("Regulatory coverage", f"{int(legal_coverage_score*100)}%")
    k4.metric("Availability risk", f"{availability_risk_score:.2f}")
    k5.metric("Fleet units", f"{int(st.session_state.get('fleet_units', st.session_state.get('req_units_avg', 1)))}")

    st.caption("Interpretation: SMR treated as fixed baseload; routing flexibility is downstream (battery/thermal/cooling).")

    # --- Baseline comparator (old version, but now ties to your computed time-to-power)
    st.subheader("Baseline Comparator (Grid + Storage)")
    grid_time_years = st.slider("Grid upgrade timeline (years)", 1.0, 10.0, 6.0, 0.5, key="ex_gridtime")
    storage_cost_proxy_crore = st.slider("Storage cost proxy (₹ crore) — reliability add-on", 0.0, 5000.0, 800.0, 50.0, key="ex_storageproxy")

    sm_p50_years = decision["ttp_p50"] / 12.0
    sm_p90_years = decision["ttp_p90"] / 12.0

    comp = pd.DataFrame({
        "Option": ["SMR co-location (P50)", "SMR co-location (P90)", "Grid upgrade"],
        "Time-to-power (years)": [sm_p50_years, sm_p90_years, grid_time_years],
        "Reliability add-on cost proxy (₹ crore)": [0.0, 0.0, storage_cost_proxy_crore]
    })
    st.dataframe(comp)

    # --- Economics highlight (only if computed)
    lcoe = st.session_state.get("lcoe_rs_per_kwh")
    implied = st.session_state.get("implied_rs_per_kwh")
    if lcoe is not None and implied is not None:
        st.subheader("Key Economic Parameter (from decomposition)")
        e1, e2, e3 = st.columns(3)
        e1.metric("LCOE (₹/kWh)", f"{float(lcoe):.2f}")
        e2.metric("Implied value (₹/kWh)", f"{float(implied):.2f}")
        e3.metric("Value − LCOE (₹/kWh)", f"{(float(implied) - float(lcoe)):.2f}")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown(
    "**Property of Sai Akshit Kurella, Jason Joel Dsilva, Muskaan Jain, Rohit Yatgiri, Shreyas Khatri**."
)
