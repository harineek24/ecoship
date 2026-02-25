"""
Fleet Intelligence Platform - Streamlit Dashboard
Carbon + Safety + Physical Workload + Workforce Retention + Cross-Impact

All data is pre-computed. The app loads CSV files and model files.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os

st.set_page_config(
    page_title="Fleet Intelligence Platform",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    [data-testid="stMetricValue"] { font-size: 1.6rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
</style>
""", unsafe_allow_html=True)


# --- Data Loading ---
@st.cache_data
def load_data():
    data = {}
    data['drivers'] = pd.read_csv('data/drivers.csv')
    data['features'] = pd.read_csv('data/weekly_features.csv')
    data['carbon'] = pd.read_csv('data/driver_carbon_summary.csv')
    data['carbon_monthly'] = pd.read_csv('data/fleet_carbon_monthly.csv')
    data['safety_scores'] = pd.read_csv('data/driver_safety_scores.csv')
    data['incidents'] = pd.read_csv('data/incident_predictions.csv')
    data['churn'] = pd.read_csv('data/churn_predictions.csv')
    data['training_roi'] = pd.read_csv('data/training_roi.csv')
    data['cross_impact'] = pd.read_csv('data/cross_impact_matrix.csv')
    data['intervention_roi'] = pd.read_csv('data/intervention_roi.csv')
    data['raw_incidents'] = pd.read_csv('data/incidents.csv')

    with open('data/whatif_coefficients.json') as f:
        data['whatif'] = json.load(f)
    with open('data/summary_stats.json') as f:
        data['summary'] = json.load(f)

    return data


@st.cache_resource
def load_models():
    models = {}
    models['carbon'] = joblib.load('models/carbon_model.joblib')
    models['incident'] = joblib.load('models/incident_model.joblib')
    models['churn'] = joblib.load('models/churn_model.joblib')
    return models


data = load_data()
models = load_models()

# --- Sidebar ---
with st.sidebar:
    st.title("Fleet Intelligence")
    st.caption("Carbon + Safety + Workforce Analytics")

    st.divider()

    region_filter = st.multiselect(
        "Filter by Region",
        ['northeast', 'southeast', 'midwest'],
        default=['northeast', 'southeast', 'midwest']
    )

    month_range = st.select_slider(
        "Month Range",
        options=list(range(1, 7)),
        value=(1, 6),
        format_func=lambda x: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'][x - 1]
    )

    st.divider()
    st.caption(f"150 drivers | 120 vehicles | 6 months")
    st.caption(f"Total CO2: {data['summary']['total_co2_kg']/1000:.0f} tons")
    st.caption(f"Total distance: {data['summary']['total_distance_km']/1e6:.1f}M km")

# Apply filters
filtered = data['features'][
    (data['features']['region'].isin(region_filter)) &
    (data['features']['month'].between(month_range[0], month_range[1]))
]

# === TABS ===
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Carbon & Emissions",
    "Safety & Risk",
    "Physical Workload",
    "Workforce & Retention",
    "Cross-Impact Analysis"
])

# =====================================================
# TAB 1 — OVERVIEW
# =====================================================
with tab1:
    st.header("Fleet Overview")

    col1, col2, col3, col4 = st.columns(4)

    total_co2 = filtered['co2_kg'].sum()
    with col1:
        st.metric(
            "Total CO2",
            f"{total_co2/1000:.1f} tons",
            delta=f"{(data['carbon_monthly'].iloc[-1]['co2_per_km'] - data['carbon_monthly'].iloc[0]['co2_per_km'])/data['carbon_monthly'].iloc[0]['co2_per_km']*100:.1f}% trend",
            delta_color="inverse"
        )

    with col2:
        safety_merged = data['safety_scores'].merge(
            filtered[['driver_id', 'week']].drop_duplicates(),
            on=['driver_id', 'week']
        )
        avg_safety = safety_merged['safety_score'].mean() if len(safety_merged) > 0 else 0
        st.metric("Fleet Safety Score", f"{avg_safety:.0f}/100")

    with col3:
        at_risk = (data['churn']['churn_prob'] > 0.5).sum()
        st.metric("Churn Risk Drivers", f"{at_risk}")

    with col4:
        total_incidents = len(data['raw_incidents'])
        st.metric("Incidents (6mo)", f"{total_incidents}")

    st.divider()

    # Fleet trends
    monthly_metrics = filtered.groupby('month').agg({
        'co2_kg': 'sum',
        'total_events_per_100km': 'mean',
        'daily_hours': 'mean'
    }).reset_index()
    monthly_metrics['month_name'] = monthly_metrics['month'].map(
        {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun'}
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(monthly_metrics, x='month_name', y='co2_kg',
                     title="Monthly Fleet CO2 Emissions (kg)",
                     labels={'co2_kg': 'CO2 (kg)', 'month_name': 'Month'},
                     color_discrete_sequence=['#27ae60'])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.line(monthly_metrics, x='month_name', y='total_events_per_100km',
                      title="Avg Driving Events per 100km",
                      labels={'total_events_per_100km': 'Events/100km',
                              'month_name': 'Month'},
                      markers=True)
        fig.update_traces(line_color='#e74c3c')
        st.plotly_chart(fig, use_container_width=True)

    # Alerts section
    st.subheader("Active Alerts")
    alerts = []

    high_risk = data['incidents'][data['incidents']['risk_score'] > 0.7]
    for _, row in high_risk.head(5).iterrows():
        factor = row['top_factor_1'].replace('_', ' ')
        alerts.append(
            f"🔴 **{row['driver_id']}** — Incident risk "
            f"{row['risk_score']:.0%} — Top factor: {factor}"
        )

    high_churn = data['churn'][data['churn']['churn_prob'] > 0.6]
    for _, row in high_churn.head(5).iterrows():
        alerts.append(
            f"🟡 **{row['driver_id']}** — Churn risk "
            f"{row['churn_prob']:.0%} — Burnout score: "
            f"{row['burnout_score']:.0f}/100"
        )

    if alerts:
        for alert in alerts[:8]:
            st.markdown(alert)
    else:
        st.info("No active alerts.")


# =====================================================
# TAB 2 — CARBON & EMISSIONS
# =====================================================
with tab2:
    st.header("Carbon & Emissions Analysis")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Fleet CO2 (6mo)", f"{total_co2/1000:.1f} tons")
    with col2:
        avg_co2_km = filtered['co2_per_km'].mean()
        st.metric("Avg CO2/km", f"{avg_co2_km:.3f} kg")
    with col3:
        idle_share = filtered['idle_fuel_pct'].mean()
        st.metric("Avg Idle Fuel %", f"{idle_share:.1f}%")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        # Emissions by source
        driving_co2 = total_co2 * (1 - idle_share / 100)
        idle_co2 = total_co2 * (idle_share / 100)
        fig = go.Figure(data=[go.Pie(
            labels=['Driving Emissions', 'Idle Emissions'],
            values=[driving_co2, idle_co2],
            hole=0.5,
            marker_colors=['#27ae60', '#e74c3c']
        )])
        fig.update_layout(title="CO2 Emissions by Source")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Driver efficiency scatter
        carbon_df = data['carbon']
        fig = px.scatter(
            carbon_df, x='total_km', y='co2_per_km',
            color='region', size='reducible_co2_kg',
            hover_data=['driver_id', 'fuel_per_100km'],
            title="Driver Efficiency vs Distance",
            labels={'total_km': 'Total Distance (km)',
                    'co2_per_km': 'CO2 per km (kg)'}
        )
        fig.add_hline(y=carbon_df['co2_per_km'].median(),
                      line_dash="dash", line_color="red",
                      annotation_text="Fleet Median")
        st.plotly_chart(fig, use_container_width=True)

    # What-If Simulator
    st.subheader("What-If Simulator")
    st.caption("Adjust sliders to estimate fleet-wide CO2 reduction")

    coef = data['whatif']
    sim_col1, sim_col2, sim_col3 = st.columns(3)

    with sim_col1:
        idle_slider = st.slider("Reduce idle time by (%)", 0, 100, 0, 5)
    with sim_col2:
        ev_slider = st.slider("Replace diesel with EV (%)", 0, 100, 0, 5)
    with sim_col3:
        coaching_slider = st.slider("Coach worst drivers (%)", 0, 50, 0, 5)

    co2_reduction = (
        idle_slider / 100 * coef['idle_coefficient'] +
        ev_slider / 100 * coef['ev_coefficient'] +
        coaching_slider / 100 * coef['coaching_coefficient']
    )
    co2_saved_tons = abs(co2_reduction) * coef['total_co2_tons']
    carbon_credit_value = co2_saved_tons * 50

    res_col1, res_col2, res_col3 = st.columns(3)
    with res_col1:
        st.metric("Estimated CO2 Reduction", f"{co2_reduction:.1%}")
    with res_col2:
        st.metric("CO2 Saved", f"{co2_saved_tons:.0f} tons/period")
    with res_col3:
        st.metric("Carbon Credit Value", f"${carbon_credit_value:,.0f}")

    st.divider()

    # SHAP plot
    st.subheader("What Drives Emissions?")
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists('results/carbon_shap_summary.png'):
            st.image('results/carbon_shap_summary.png',
                     caption="SHAP Feature Importance for CO2/km Model")
    with col2:
        if os.path.exists('results/carbon_by_behavior.png'):
            st.image('results/carbon_by_behavior.png',
                     caption="Top Behaviors Driving Emissions")

    # ESG summary
    st.subheader("ESG Monthly Summary")
    st.dataframe(data['carbon_monthly'], use_container_width=True)
    st.download_button(
        "Download ESG Report (CSV)",
        data['carbon_monthly'].to_csv(index=False),
        "fleet_carbon_monthly.csv",
        "text/csv"
    )


# =====================================================
# TAB 3 — SAFETY & RISK
# =====================================================
with tab3:
    st.header("Safety & Risk Analysis")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Fleet Avg Safety Score", f"{avg_safety:.0f}/100")
    with col2:
        high_risk_count = (data['incidents']['risk_score'] > 0.5).sum()
        st.metric("High-Risk Drivers", f"{high_risk_count}")
    with col3:
        st.metric("Total Incidents", f"{len(data['raw_incidents'])}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        # Safety score distribution
        if os.path.exists('results/safety_score_distribution.png'):
            st.image('results/safety_score_distribution.png',
                     caption="Fleet Safety Score Distribution")

    with col2:
        # ROC curve
        if os.path.exists('results/incident_roc.png'):
            st.image('results/incident_roc.png',
                     caption="Incident Prediction Model Performance")

    # At-risk drivers table
    st.subheader("Top At-Risk Drivers")
    risk_df = data['incidents'].head(15).copy()
    risk_df['risk_score'] = risk_df['risk_score'].apply(lambda x: f"{x:.0%}")
    risk_df.columns = [c.replace('_', ' ').title() for c in risk_df.columns]
    st.dataframe(risk_df, use_container_width=True, hide_index=True)

    # Drill-down per driver
    st.subheader("Driver Safety Detail")
    selected_driver = st.selectbox(
        "Select a driver",
        data['incidents']['driver_id'].tolist(),
        key='safety_driver_select'
    )

    if selected_driver:
        with st.expander(f"Details for {selected_driver}", expanded=True):
            driver_scores = data['safety_scores'][
                data['safety_scores']['driver_id'] == selected_driver
            ].sort_values('week')

            fig = px.line(driver_scores, x='week', y='safety_score',
                          title=f"Safety Score Trend — {selected_driver}",
                          markers=True)
            fig.update_traces(line_color='#2980b9')
            fig.add_hline(y=avg_safety, line_dash="dash", line_color="gray",
                          annotation_text="Fleet Avg")
            st.plotly_chart(fig, use_container_width=True)

            driver_pred = data['incidents'][
                data['incidents']['driver_id'] == selected_driver
            ].iloc[0]
            st.markdown(f"**Risk Score:** {driver_pred['risk_score']:.0%}")
            st.markdown(f"**Top Risk Factors:**")
            st.markdown(f"1. {driver_pred['top_factor_1'].replace('_', ' ')}")
            st.markdown(f"2. {driver_pred['top_factor_2'].replace('_', ' ')}")
            st.markdown(f"3. {driver_pred['top_factor_3'].replace('_', ' ')}")

    # SHAP summary
    st.subheader("What Predicts Incidents?")
    if os.path.exists('results/safety_shap_summary.png'):
        st.image('results/safety_shap_summary.png',
                 caption="SHAP Feature Importance for Incident Prediction")


# =====================================================
# TAB 4 — PHYSICAL WORKLOAD
# =====================================================
with tab4:
    st.header("Physical Workload Analysis")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Daily Steps", f"{filtered['daily_steps'].mean():,.0f}")
    with col2:
        st.metric("Avg Daily Floors", f"{filtered['daily_floors'].mean():.1f}")
    with col3:
        st.metric("Avg Package Weight", f"{filtered['avg_package_weight_kg'].mean():.1f} kg")

    st.divider()

    # Box plots by region
    col1, col2 = st.columns(2)

    with col1:
        fig = px.box(filtered, x='region', y='daily_steps',
                     title="Daily Steps by Region",
                     color='region',
                     labels={'daily_steps': 'Daily Steps', 'region': 'Region'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(filtered, x='region', y='daily_floors',
                     title="Daily Floors by Region",
                     color='region',
                     labels={'daily_floors': 'Daily Floors', 'region': 'Region'})
        st.plotly_chart(fig, use_container_width=True)

    # Physical workload fairness
    st.subheader("Workload Equity")

    p80_steps = filtered['daily_steps'].quantile(0.8)
    p80_floors = filtered['daily_floors'].quantile(0.8)
    p80_lifting = filtered['daily_lifting_events'].quantile(0.8)

    high_load = filtered[
        (filtered['daily_steps'] > p80_steps) |
        (filtered['daily_floors'] > p80_floors) |
        (filtered['daily_lifting_events'] > p80_lifting)
    ].groupby('driver_id').agg({
        'daily_steps': 'mean',
        'daily_floors': 'mean',
        'daily_lifting_events': 'mean',
        'region': 'first',
    }).reset_index().sort_values('daily_steps', ascending=False)

    st.markdown(f"**{len(high_load)} drivers** have physical workload above 80th percentile")
    st.dataframe(high_load.head(15), use_container_width=True, hide_index=True)

    # Scatter: physical exertion vs safety
    st.subheader("Physical Load vs Safety Performance")
    driver_avg = filtered.groupby('driver_id').agg({
        'daily_steps': 'mean',
        'daily_floors': 'mean',
        'total_events_per_100km': 'mean',
        'region': 'first',
    }).reset_index()

    fig = px.scatter(driver_avg, x='daily_steps', y='total_events_per_100km',
                     color='region',
                     title="Physical Activity vs Safety Events",
                     labels={
                         'daily_steps': 'Avg Daily Steps',
                         'total_events_per_100km': 'Safety Events per 100km'
                     },
                     trendline='ols')
    st.plotly_chart(fig, use_container_width=True)


# =====================================================
# TAB 5 — WORKFORCE & RETENTION
# =====================================================
with tab5:
    st.header("Workforce & Retention")

    col1, col2, col3 = st.columns(3)
    with col1:
        churned = data['drivers']['termination_date'].notna().sum()
        st.metric("Churned Drivers (6mo)", f"{churned}")
    with col2:
        churn_risk = (data['churn']['churn_prob'] > 0.5).sum()
        st.metric("Current Churn Risk", f"{churn_risk} drivers")
    with col3:
        avg_burnout = data['churn']['burnout_score'].mean()
        st.metric("Avg Burnout Score", f"{avg_burnout:.0f}/100")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        # Churn risk ranked list
        st.subheader("Churn Risk Rankings")
        churn_display = data['churn'].head(20).copy()
        churn_display['churn_prob'] = churn_display['churn_prob'].apply(
            lambda x: f"{x:.0%}"
        )
        churn_display['burnout_score'] = churn_display['burnout_score'].apply(
            lambda x: f"{x:.0f}"
        )
        churn_display.columns = [c.replace('_', ' ').title() for c in churn_display.columns]
        st.dataframe(churn_display, use_container_width=True, hide_index=True)

    with col2:
        # Survival curves
        if os.path.exists('results/survival_curves.png'):
            st.image('results/survival_curves.png',
                     caption="Kaplan-Meier Driver Retention Curves by Region")

    # Burnout scatter
    st.subheader("Burnout vs Safety")
    if os.path.exists('results/burnout_scatter.png'):
        st.image('results/burnout_scatter.png',
                 caption="Burnout Risk (color) vs Daily Hours, sized by churn probability")

    # Training ROI
    st.subheader("Training ROI")
    col1, col2 = st.columns(2)

    with col1:
        if os.path.exists('results/training_impact.png'):
            st.image('results/training_impact.png',
                     caption="Training Before/After Comparison")

    with col2:
        st.dataframe(data['training_roi'], use_container_width=True, hide_index=True)

    # Retention cost calculator
    st.subheader("Retention Cost Calculator")
    hire_cost = st.number_input(
        "Cost to hire replacement ($)", value=8000, step=1000
    )

    predicted_churns = (data['churn']['churn_prob'] > 0.5).sum()
    total_exposure = predicted_churns * hire_cost
    coaching_cost_per_driver = 500
    coaching_total = predicted_churns * coaching_cost_per_driver
    retention_rate = 0.4
    saved_churns = int(predicted_churns * retention_rate)
    savings = saved_churns * hire_cost

    calc_col1, calc_col2, calc_col3 = st.columns(3)
    with calc_col1:
        st.metric("Predicted Churns", f"{predicted_churns}")
        st.metric("Total Exposure", f"${total_exposure:,}")
    with calc_col2:
        st.metric("Coaching Cost", f"${coaching_total:,}")
        st.metric("Saved Churns (est.)", f"{saved_churns}")
    with calc_col3:
        st.metric("Retention Savings", f"${savings:,}")
        net = savings - coaching_total
        st.metric("Net Savings", f"${net:,}")


# =====================================================
# TAB 6 — CROSS-IMPACT ANALYSIS
# =====================================================
with tab6:
    st.header("Cross-Impact Analysis")

    st.info(
        "**Key Finding:** Eco-driving coaching improves ALL outcomes: "
        "-18% emissions, -12% incidents, +8% retention. "
        "ROI: 3.6x. This is the single best multi-outcome investment."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        # Cross-correlation heatmap
        st.subheader("Metric Correlations")
        if os.path.exists('results/cross_correlation_heatmap.png'):
            st.image('results/cross_correlation_heatmap.png',
                     caption="How carbon, safety, and workload metrics interact")

    with col2:
        # Intervention comparison
        st.subheader("Intervention Impact Comparison")
        if os.path.exists('results/intervention_comparison.png'):
            st.image('results/intervention_comparison.png',
                     caption="Each intervention's effect across all outcome areas")

    # Interactive intervention comparison
    st.subheader("Intervention ROI Comparison")
    impact_df = data['cross_impact']

    fig = go.Figure()
    categories = ['carbon', 'safety', 'retention', 'injury']
    colors = ['#27ae60', '#2980b9', '#e67e22', '#c0392b']

    for cat, color in zip(categories, colors):
        fig.add_trace(go.Bar(
            name=cat.capitalize(),
            x=impact_df['intervention'],
            y=impact_df[cat],
            marker_color=color
        ))

    fig.update_layout(
        barmode='group',
        title="Cross-Impact: How Each Intervention Affects All Outcomes",
        yaxis_title="Impact (%)",
        xaxis_title="Intervention",
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ROI table
    st.subheader("Full Cost-Benefit Analysis")
    roi_df = data['intervention_roi'][[
        'intervention', 'description', 'annual_cost',
        'carbon_savings_usd', 'incident_savings_usd',
        'retention_savings_usd', 'total_savings_usd', 'net_roi'
    ]].copy()
    roi_df.columns = [
        'Intervention', 'Description', 'Annual Cost ($)',
        'Carbon Savings ($)', 'Incident Savings ($)',
        'Retention Savings ($)', 'Total Savings ($)', 'ROI (x)'
    ]
    roi_df = roi_df.sort_values('ROI (x)', ascending=False)
    st.dataframe(roi_df, use_container_width=True, hide_index=True)

    # Download executive summary
    st.divider()
    exec_summary = (
        "FLEET INTELLIGENCE PLATFORM - EXECUTIVE SUMMARY\n"
        "================================================\n\n"
        f"Fleet: 150 drivers, 120 vehicles, 3 regions\n"
        f"Period: Jan-Jun 2024 (6 months)\n\n"
        f"CARBON: {total_co2/1000:.0f} tons CO2 total, "
        f"{avg_co2_km:.3f} kg/km average\n"
        f"SAFETY: {avg_safety:.0f}/100 fleet safety score, "
        f"{len(data['raw_incidents'])} incidents\n"
        f"RETENTION: {churned} drivers churned, "
        f"{at_risk} currently at risk\n\n"
        "TOP INTERVENTION RECOMMENDATIONS:\n"
    )
    for _, row in roi_df.iterrows():
        exec_summary += (
            f"  {row['Intervention']}: "
            f"Cost ${row['Annual Cost ($)']:,}, "
            f"Saves ${row['Total Savings ($)']:,}, "
            f"ROI {row['ROI (x)']}x\n"
        )

    st.download_button(
        "Download Executive Summary",
        exec_summary,
        "fleet_intelligence_summary.txt",
        "text/plain"
    )
