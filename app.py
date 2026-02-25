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
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        flex-wrap: wrap;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.85rem;
        padding: 6px 12px;
    }
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

    # Advanced ML artifacts
    if os.path.exists('data/anomaly_details.csv'):
        data['anomaly_details'] = pd.read_csv('data/anomaly_details.csv')
        data['anomaly_scores'] = pd.read_csv('data/anomaly_scores.csv')
    if os.path.exists('data/driver_clusters.csv'):
        data['clusters'] = pd.read_csv('data/driver_clusters.csv')
    if os.path.exists('data/cluster_profiles.csv'):
        data['cluster_profiles'] = pd.read_csv('data/cluster_profiles.csv')
    if os.path.exists('data/cluster_info.json'):
        with open('data/cluster_info.json') as f:
            data['cluster_info'] = json.load(f)
    if os.path.exists('data/cox_hazard_ratios.csv'):
        data['cox_hr'] = pd.read_csv('data/cox_hazard_ratios.csv')
    if os.path.exists('data/cox_info.json'):
        with open('data/cox_info.json') as f:
            data['cox_info'] = json.load(f)
    if os.path.exists('data/ensemble_comparison.csv'):
        data['ensemble'] = pd.read_csv('data/ensemble_comparison.csv')
    if os.path.exists('data/tuning_results.json'):
        with open('data/tuning_results.json') as f:
            data['tuning'] = json.load(f)
    if os.path.exists('data/tuning_history.csv'):
        data['tuning_history'] = pd.read_csv('data/tuning_history.csv')
    if os.path.exists('data/gru_results.json'):
        with open('data/gru_results.json') as f:
            data['gru'] = json.load(f)
    if os.path.exists('data/gru_training_history.csv'):
        data['gru_history'] = pd.read_csv('data/gru_training_history.csv')
    if os.path.exists('data/gru_attention.csv'):
        data['gru_attention'] = pd.read_csv('data/gru_attention.csv')

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
    n_drivers = data['summary'].get('n_drivers', 150)
    st.caption(f"{n_drivers} drivers | {int(n_drivers*0.8)} vehicles | 6 months")
    st.caption(f"Total CO2: {data['summary']['total_co2_kg']/1000:.0f} tons")
    st.caption(f"Total distance: {data['summary']['total_distance_km']/1e6:.1f}M km")

# Apply filters
filtered = data['features'][
    (data['features']['region'].isin(region_filter)) &
    (data['features']['month'].between(month_range[0], month_range[1]))
]

# Pre-compute shared metrics
total_co2 = filtered['co2_kg'].sum()
avg_co2_km = filtered['co2_per_km'].mean()
safety_merged = data['safety_scores'].merge(
    filtered[['driver_id', 'week']].drop_duplicates(),
    on=['driver_id', 'week']
)
avg_safety = safety_merged['safety_score'].mean() if len(safety_merged) > 0 else 0
at_risk = (data['churn']['churn_prob'] > 0.5).sum()
churned = data['drivers']['termination_date'].notna().sum()
idle_share = filtered['idle_fuel_pct'].mean()

# === TABS ===
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Overview",
    "Carbon",
    "Safety",
    "Workload",
    "Workforce",
    "Cross-Impact",
    "Advanced ML",
    "How It Works"
])

# =====================================================
# TAB 1 — OVERVIEW
# =====================================================
with tab1:
    st.header("Fleet Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        co2_trend = (data['carbon_monthly'].iloc[-1]['co2_per_km'] - data['carbon_monthly'].iloc[0]['co2_per_km']) / max(data['carbon_monthly'].iloc[0]['co2_per_km'], 1e-9) * 100
        st.metric(
            "Total CO2",
            f"{total_co2/1000:.1f} tons",
            delta=f"{co2_trend:.1f}% vs Jan",
            delta_color="inverse"
        )

    with col2:
        # Compute early vs late safety to show trend
        early_safety = data['safety_scores'][data['safety_scores']['week'] <= 13]['safety_score'].mean()
        late_safety = data['safety_scores'][data['safety_scores']['week'] > 13]['safety_score'].mean()
        safety_delta = late_safety - early_safety
        st.metric("Fleet Safety Score", f"{avg_safety:.0f}/100",
                  delta=f"{safety_delta:+.1f} pts H2 vs H1")

    with col3:
        st.metric("Churn Risk Drivers", f"{at_risk}",
                  delta=f"{at_risk} need intervention",
                  delta_color="inverse")

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
        st.metric("Avg CO2/km", f"{avg_co2_km:.3f} kg")
    with col3:
        st.metric("Avg Idle Fuel %", f"{idle_share:.1f}%")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        # Emissions by source donut
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
            title="Driver Efficiency vs Distance (size = reducible CO2)",
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

    # SHAP plots
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
        # Interactive safety score histogram
        fig = px.histogram(
            safety_merged, x='safety_score', nbins=30,
            title="Fleet Safety Score Distribution",
            labels={'safety_score': 'Safety Score', 'count': 'Count'},
            color_discrete_sequence=['steelblue']
        )
        fig.add_vline(x=avg_safety, line_dash="dash", line_color="red",
                      annotation_text=f"Mean: {avg_safety:.0f}")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # ROC curve
        if os.path.exists('results/incident_roc.png'):
            st.image('results/incident_roc.png',
                     caption="Incident Prediction Model Performance")

    # At-risk drivers table
    st.subheader("Top At-Risk Drivers")
    risk_df = data['incidents'].head(15).copy()
    risk_df['risk_score_fmt'] = risk_df['risk_score'].apply(lambda x: f"{x:.0%}")
    display_risk = risk_df[['driver_id', 'risk_score_fmt', 'top_factor_1', 'top_factor_2', 'top_factor_3']].copy()
    display_risk.columns = ['Driver', 'Risk Score', 'Top Factor 1', 'Top Factor 2', 'Top Factor 3']
    for col in ['Top Factor 1', 'Top Factor 2', 'Top Factor 3']:
        display_risk[col] = display_risk[col].str.replace('_', ' ')
    st.dataframe(display_risk, use_container_width=True, hide_index=True)

    # Drill-down per driver
    st.subheader("Driver Safety Detail")
    selected_driver = st.selectbox(
        "Select a driver to inspect",
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

            # Risk factors as a bar chart (SHAP-like waterfall)
            factors = {
                driver_pred['top_factor_1'].replace('_', ' '): 0.35,
                driver_pred['top_factor_2'].replace('_', ' '): 0.25,
                driver_pred['top_factor_3'].replace('_', ' '): 0.15,
            }

            # Get the driver's actual feature values for context
            driver_features = filtered[filtered['driver_id'] == selected_driver]
            if len(driver_features) > 0:
                latest = driver_features.iloc[-1]
                raw_factor_1 = driver_pred['top_factor_1']
                if raw_factor_1 in latest.index:
                    fleet_median = filtered[raw_factor_1].median()
                    driver_val = latest[raw_factor_1]
                    st.markdown(f"**Risk Score:** {driver_pred['risk_score']:.0%}")
                    st.markdown("**Top Risk Factors (with driver vs fleet comparison):**")

                    for i, factor_col in enumerate([driver_pred['top_factor_1'],
                                                     driver_pred['top_factor_2'],
                                                     driver_pred['top_factor_3']]):
                        if factor_col in latest.index:
                            d_val = latest[factor_col]
                            f_med = filtered[factor_col].median()
                            direction = "above" if d_val > f_med else "below"
                            pct_diff = abs(d_val - f_med) / max(abs(f_med), 1e-9) * 100
                            st.markdown(
                                f"{i+1}. **{factor_col.replace('_', ' ')}**: "
                                f"{d_val:.2f} ({pct_diff:.0f}% {direction} fleet median of {f_med:.2f})"
                            )
                else:
                    st.markdown(f"**Risk Score:** {driver_pred['risk_score']:.0%}")
            else:
                st.markdown(f"**Risk Score:** {driver_pred['risk_score']:.0%}")
                st.markdown(f"1. {driver_pred['top_factor_1'].replace('_', ' ')}")
                st.markdown(f"2. {driver_pred['top_factor_2'].replace('_', ' ')}")
                st.markdown(f"3. {driver_pred['top_factor_3'].replace('_', ' ')}")

            # Coaching recommendation
            st.markdown("---")
            st.markdown("**Recommended Action:**")
            top_factor = driver_pred['top_factor_1']
            coaching_map = {
                'total_events_per_100km': 'Enroll in comprehensive defensive driving course. This driver has elevated event rates across multiple categories.',
                'hard_brakes_per_100km': 'Focus on following distance training. Hard braking usually indicates tailgating or late reaction to traffic.',
                'speeding_per_100km': 'Implement speed governor or GPS-based speed alerts. Review routes for speed limit awareness.',
                'phone_use_per_100km': 'Install phone-lock-while-driving technology. Mandatory distracted driving awareness session.',
                'night_driving_pct': 'Review shift schedule — high night driving correlates with fatigue-related incidents. Consider rotating to day shifts.',
                'hard_accels_per_100km': 'Eco-driving coaching focused on smooth acceleration. Often co-occurs with aggressive driving patterns.',
                'idle_time_pct': 'Anti-idle coaching and route optimization. Excessive idling suggests poor route planning or engine-on habits during deliveries.',
                'daily_hours': 'Workload review — extended hours increase fatigue risk. Consider route rebalancing to reduce daily driving time.',
                'experience_years': 'Pair with experienced mentor driver. New drivers benefit from ride-alongs during first 6 months.',
                'age': 'Ergonomic assessment and adjusted delivery routes. Physical workload accommodation may reduce injury risk.',
                'safety_score': 'Comprehensive safety review — multiple contributing factors. Schedule 1-on-1 coaching session.',
                'safety_score_trend': 'Declining safety trend detected. Immediate supervisor check-in recommended to identify root cause.',
            }
            recommendation = coaching_map.get(
                top_factor,
                f'Review {top_factor.replace("_", " ")} patterns and schedule targeted coaching session.'
            )
            st.info(recommendation)

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

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Daily Steps", f"{filtered['daily_steps'].mean():,.0f}")
    with col2:
        st.metric("Avg Daily Floors", f"{filtered['daily_floors'].mean():.1f}")
    with col3:
        st.metric("Avg Lifting Events/Day", f"{filtered['daily_lifting_events'].mean():.1f}")
    with col4:
        st.metric("Avg Package Weight", f"{filtered['avg_package_weight_kg'].mean():.1f} kg")

    st.divider()

    # Box plots by region — steps, floors, AND lifting
    col1, col2, col3 = st.columns(3)

    with col1:
        fig = px.box(filtered, x='region', y='daily_steps',
                     title="Daily Steps by Region",
                     color='region',
                     labels={'daily_steps': 'Daily Steps', 'region': 'Region'})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(filtered, x='region', y='daily_floors',
                     title="Daily Floors by Region",
                     color='region',
                     labels={'daily_floors': 'Daily Floors', 'region': 'Region'})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        fig = px.box(filtered, x='region', y='daily_lifting_events',
                     title="Daily Lifting Events by Region",
                     color='region',
                     labels={'daily_lifting_events': 'Lifting Events', 'region': 'Region'})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Fatigue indicator: morning vs afternoon performance
    st.subheader("Fatigue Indicator")
    st.caption("Comparing morning shift vs night shift drivers — night/rotating shifts show higher event rates, suggesting fatigue effects")

    shift_fatigue = filtered.groupby('shift').agg({
        'total_events_per_100km': 'mean',
        'hard_brakes_per_100km': 'mean',
        'speeding_per_100km': 'mean',
        'daily_steps': 'mean',
    }).reset_index()

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(shift_fatigue, x='shift', y='total_events_per_100km',
                     title="Safety Events by Shift Type",
                     color='shift',
                     labels={'total_events_per_100km': 'Events/100km', 'shift': 'Shift'})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fatigue_metrics = shift_fatigue.melt(
            id_vars='shift',
            value_vars=['hard_brakes_per_100km', 'speeding_per_100km'],
            var_name='metric', value_name='rate'
        )
        fatigue_metrics['metric'] = fatigue_metrics['metric'].map({
            'hard_brakes_per_100km': 'Hard Brakes',
            'speeding_per_100km': 'Speeding',
        })
        fig = px.bar(fatigue_metrics, x='shift', y='rate', color='metric',
                     title="Fatigue-Related Events by Shift",
                     barmode='group',
                     labels={'rate': 'Events per 100km', 'shift': 'Shift'})
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
                     })
    st.plotly_chart(fig, use_container_width=True)


# =====================================================
# TAB 5 — WORKFORCE & RETENTION
# =====================================================
with tab5:
    st.header("Workforce & Retention")

    col1, col2, col3 = st.columns(3)
    with col1:
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
        churn_display['churn_prob_fmt'] = churn_display['churn_prob'].apply(lambda x: f"{x:.0%}")
        churn_display['burnout_fmt'] = churn_display['burnout_score'].apply(lambda x: f"{x:.0f}")
        display_churn = churn_display[['driver_id', 'churn_prob_fmt', 'burnout_fmt',
                                        'top_factor_1', 'top_factor_2']].copy()
        display_churn.columns = ['Driver', 'Churn Prob', 'Burnout', 'Top Factor 1', 'Top Factor 2']
        for col in ['Top Factor 1', 'Top Factor 2']:
            display_churn[col] = display_churn[col].str.replace('_', ' ')
        st.dataframe(display_churn, use_container_width=True, hide_index=True)

    with col2:
        # Survival curves
        if os.path.exists('results/survival_curves.png'):
            st.image('results/survival_curves.png',
                     caption="Kaplan-Meier Driver Retention Curves by Region")

    # Interactive Burnout scatter (replaces static PNG)
    st.subheader("Burnout Risk Map")
    st.caption("Each dot is a driver. Size = churn probability, color = burnout score.")

    # Build per-driver latest data for scatter
    churn_df = data['churn'].copy()
    # Merge with features to get daily_hours and safety_score
    latest_features = filtered.groupby('driver_id').tail(1)[['driver_id', 'daily_hours']].copy()
    safety_latest = data['safety_scores'].groupby('driver_id').tail(1)[['driver_id', 'safety_score']].copy()

    burnout_plot = churn_df.merge(latest_features, on='driver_id', how='left')
    burnout_plot = burnout_plot.merge(safety_latest, on='driver_id', how='left')
    burnout_plot = burnout_plot.dropna(subset=['daily_hours', 'safety_score'])

    if len(burnout_plot) > 0:
        fig = px.scatter(
            burnout_plot,
            x='daily_hours',
            y='safety_score',
            color='burnout_score',
            size='churn_prob',
            hover_data=['driver_id', 'churn_prob', 'burnout_score'],
            title="Driver Burnout: Daily Hours vs Safety Score",
            labels={
                'daily_hours': 'Daily Hours',
                'safety_score': 'Safety Score',
                'burnout_score': 'Burnout Score',
                'churn_prob': 'Churn Probability'
            },
            color_continuous_scale='RdYlGn_r',
            size_max=25
        )
        fig.update_layout(coloraxis_colorbar_title="Burnout")
        st.plotly_chart(fig, use_container_width=True)

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
        st.subheader("Metric Correlations")
        if os.path.exists('results/cross_correlation_heatmap.png'):
            st.image('results/cross_correlation_heatmap.png',
                     caption="How carbon, safety, and workload metrics interact")

    with col2:
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
        f"Fleet: 2,000 drivers, 1,600 vehicles, 3 regions\n"
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


# =====================================================
# TAB 7 — ADVANCED ML
# =====================================================
with tab7:
    st.header("Advanced ML Techniques")
    st.caption("Anomaly detection, unsupervised clustering, survival regression, "
               "model comparison, and hyperparameter optimization.")

    st.divider()

    # --- 1. ANOMALY DETECTION ---
    st.subheader("1. Anomaly Detection (Isolation Forest)")
    st.markdown("""
An Isolation Forest identifies driver-weeks with **unusual multivariate behavior patterns** that
simple thresholds would miss. It works by randomly partitioning data — anomalies are isolated
in fewer splits because they're far from the norm in multiple dimensions simultaneously.
""")

    if 'anomaly_details' in data:
        anomaly_df = data['anomaly_details']
        col1, col2 = st.columns(2)
        with col1:
            n_anom = len(anomaly_df)
            n_drivers = anomaly_df['driver_id'].nunique()
            total_weeks = len(data['anomaly_scores'])
            st.metric("Anomalous Weeks Detected", f"{n_anom} / {total_weeks}")
            st.metric("Drivers with Anomalies", f"{n_drivers}")

            if os.path.exists('results/anomaly_distribution.png'):
                st.image('results/anomaly_distribution.png',
                         caption="Anomaly score distribution — threshold marks the 5% contamination boundary")

        with col2:
            st.markdown("**Top Anomalous Driver-Weeks**")
            display_anom = anomaly_df.sort_values('anomaly_score').head(15).copy()
            display_anom['reason_1'] = display_anom['reason_1'].str.replace('_', ' ')
            display_anom['reason_2'] = display_anom['reason_2'].str.replace('_', ' ')
            display_anom.columns = ['Driver', 'Week', 'Score', 'Reason 1', 'Z1',
                                     'Reason 2', 'Z2', 'Reason 3', 'Z3']
            st.dataframe(
                display_anom[['Driver', 'Week', 'Score', 'Reason 1', 'Z1', 'Reason 2', 'Z2']],
                use_container_width=True, hide_index=True
            )
    else:
        st.info("Run the advanced models pipeline to generate anomaly detection results.")

    st.divider()

    # --- 2. DRIVER CLUSTERING ---
    st.subheader("2. Driver Clustering (K-Means + PCA)")
    st.markdown("""
K-Means discovers natural **driver behavior profiles** from the data. The hidden `_driving_style`
variable (cautious / normal / aggressive / fatigued) is never exposed to the model — so the
clusters it finds are entirely data-driven. PCA reduces the 14 features to 2D for visualization.
""")

    if 'clusters' in data:
        cluster_df = data['clusters']
        cluster_info = data.get('cluster_info', {})
        optimal_k = cluster_info.get('optimal_k', cluster_df['cluster'].nunique())
        sil_score = cluster_info.get('silhouette_score', 0)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Clusters Discovered", f"{optimal_k}")
            st.metric("Silhouette Score", f"{sil_score:.3f}")

            if os.path.exists('results/silhouette_scores.png'):
                st.image('results/silhouette_scores.png',
                         caption="Silhouette score by k — higher is better separation")

        with col2:
            if os.path.exists('results/cluster_pca.png'):
                st.image('results/cluster_pca.png',
                         caption="PCA projection of driver clusters")

        # Interactive cluster scatter
        if 'pca_1' in cluster_df.columns:
            fig = px.scatter(
                cluster_df, x='pca_1', y='pca_2',
                color=cluster_df['cluster'].astype(str),
                hover_data=['driver_id', 'fuel_per_100km',
                            'total_events_per_100km', 'daily_hours'],
                title="Driver Clusters (Interactive PCA)",
                labels={'pca_1': 'PC1', 'pca_2': 'PC2', 'color': 'Cluster'}
            )
            fig.update_layout(legend_title_text='Cluster')
            st.plotly_chart(fig, use_container_width=True)

        # Cluster profiles
        if 'cluster_profiles' in data:
            st.markdown("**Cluster Profiles (mean feature values)**")
            profiles = data['cluster_profiles']
            display_cols = ['cluster', 'hard_brakes_per_100km', 'fuel_per_100km',
                            'total_events_per_100km', 'daily_hours', 'co2_per_km',
                            'daily_steps']
            avail_cols = [c for c in display_cols if c in profiles.columns]
            st.dataframe(profiles[avail_cols], use_container_width=True, hide_index=True)
    else:
        st.info("Run the advanced models pipeline to generate clustering results.")

    st.divider()

    # --- 3. COX PROPORTIONAL HAZARDS ---
    st.subheader("3. Cox Proportional Hazards (Survival Regression)")
    st.markdown("""
While Kaplan-Meier curves (Workforce tab) show *descriptive* retention over time, Cox PH is a
**regression model** that quantifies which features accelerate or delay churn. A hazard ratio (HR)
of 1.5 means a 1-SD increase in that feature increases churn risk by 50%.
""")

    if 'cox_hr' in data:
        cox_hr = data['cox_hr']
        cox_info = data.get('cox_info', {})

        col1, col2 = st.columns(2)
        with col1:
            ci = cox_info.get('concordance_index')
            if ci:
                st.metric("Concordance Index", f"{ci:.3f}")
            st.metric("Subjects", f"{cox_info.get('n_subjects', 'N/A')}")
            st.metric("Events (churns)", f"{cox_info.get('n_events', 'N/A')}")

        with col2:
            if os.path.exists('results/cox_hazard_ratios.png'):
                st.image('results/cox_hazard_ratios.png',
                         caption="Hazard ratios — red accelerates churn, green is protective")

        # Interactive hazard ratio chart
        fig = go.Figure()
        cox_sorted = cox_hr.sort_values('hazard_ratio')
        colors = ['#e74c3c' if hr > 1 else '#27ae60'
                  for hr in cox_sorted['hazard_ratio']]
        fig.add_trace(go.Bar(
            y=[f.replace('_', ' ') for f in cox_sorted['feature']],
            x=cox_sorted['hazard_ratio'],
            orientation='h',
            marker_color=colors,
        ))
        fig.add_vline(x=1, line_dash="solid", line_color="black")
        fig.update_layout(
            title="Hazard Ratios (HR > 1 increases churn risk)",
            xaxis_title="Hazard Ratio",
            showlegend=False,
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run the advanced models pipeline to generate Cox PH results.")

    st.divider()

    # --- 4. STACKED ENSEMBLE + MODEL COMPARISON ---
    st.subheader("4. Model Comparison + Stacked Ensemble")
    st.markdown("""
Instead of only XGBoost, we train **4 different algorithms** on the same tasks and compare:
- **XGBoost** — gradient boosting with tree-based splits
- **LightGBM** — histogram-based gradient boosting (faster, different bias)
- **Random Forest** — bagged decision trees (less prone to overfitting)
- **Logistic Regression** — linear baseline (interpretable, regularized)

A **stacked ensemble** feeds all 4 models' predictions into a meta-learner (Logistic Regression)
to see if combining them improves performance.
""")

    if 'ensemble' in data:
        ensemble_df = data['ensemble']

        for _, row in ensemble_df.iterrows():
            task = row['task'].title()
            st.markdown(f"**{task} Prediction — AUC Comparison**")

            model_names = ['XGBoost', 'LightGBM', 'Random Forest',
                           'Logistic Regression', 'Stacked Ensemble']
            aucs = [
                row['xgboost_auc'], row['lightgbm_auc'],
                row['random_forest_auc'], row['logistic_regression_auc'],
                row['stacked_ensemble_auc']
            ]
            best_auc = max(aucs)
            colors = ['#e74c3c' if a == best_auc else '#3498db' for a in aucs]

            fig = go.Figure(data=[go.Bar(
                x=model_names, y=aucs,
                marker_color=colors,
                text=[f'{a:.3f}' for a in aucs],
                textposition='outside'
            )])
            fig.update_layout(
                yaxis_title="AUC-ROC",
                yaxis_range=[min(aucs) * 0.9, max(aucs) * 1.05],
                height=350,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Show ROC curve comparison
            roc_path = f'results/ensemble_roc_{row["task"]}.png'
            if os.path.exists(roc_path):
                st.image(roc_path,
                         caption=f"{task}: ROC curves for all models")
    else:
        st.info("Run the advanced models pipeline to generate ensemble comparison.")

    st.divider()

    # --- 5. HYPERPARAMETER TUNING ---
    st.subheader("5. Hyperparameter Tuning (Optuna + Time-Series CV)")
    st.markdown("""
The original models used hardcoded hyperparameters (`max_depth=5, learning_rate=0.05`).
**Optuna** performs Bayesian optimization — it samples hyperparameter combinations intelligently
(not random grid search) to find the best config in fewer trials.

**Time-series CV** prevents data leakage: each fold uses an expanding training window
(weeks 1-8, 1-12, 1-16) with a forward-looking validation window. This respects the temporal
ordering of the data.
""")

    if 'tuning' in data:
        tuning = data['tuning']

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Default AUC", f"{tuning['default_auc']:.4f}")
        with col2:
            st.metric("Tuned AUC", f"{tuning['tuned_auc']:.4f}")
        with col3:
            st.metric("Improvement",
                       f"+{tuning['improvement_pct']:.1f}%",
                       delta=f"{tuning['improvement_pct']:.1f}%")

        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists('results/tuning_history.png'):
                st.image('results/tuning_history.png',
                         caption="Left: optimization convergence. "
                                 "Right: which hyperparameters matter most.")
        with col2:
            if os.path.exists('results/tuning_roc_comparison.png'):
                st.image('results/tuning_roc_comparison.png',
                         caption="Default vs tuned XGBoost on held-out test set")

        # Show best parameters
        st.markdown("**Best Hyperparameters Found**")
        best_params = tuning.get('best_params', {})
        params_df = pd.DataFrame([
            {'Parameter': k.replace('_', ' ').title(),
             'Value': f"{v:.4f}" if isinstance(v, float) else str(v)}
            for k, v in best_params.items()
        ])
        st.dataframe(params_df, use_container_width=True, hide_index=True)

        # Tuning history scatter
        if 'tuning_history' in data:
            history = data['tuning_history']
            fig = px.scatter(
                history, x='trial', y='auc_cv',
                title="Optuna Trial History (50 trials)",
                labels={'trial': 'Trial Number', 'auc_cv': 'AUC (CV)'},
                color='auc_cv',
                color_continuous_scale='RdYlGn',
            )
            fig.update_layout(coloraxis_colorbar_title="AUC")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run the advanced models pipeline to generate tuning results.")

    st.divider()

    # --- 6. SEQUENCE MODEL (GRU) ---
    st.subheader("6. Sequence Model (Bidirectional GRU + Attention)")
    st.markdown("""
XGBoost treats each driver-week as an **independent row** — it can't see the trajectory.
A GRU (Gated Recurrent Unit) takes the full weekly time series for each driver and learns
sequential patterns: behavior deterioration over weeks, burnout trajectories, fatigue accumulation.

**Architecture:**
- **Bidirectional GRU** (2 layers, hidden=64) — reads the sequence forwards and backwards
- **Attention pooling** — learns which weeks matter most (instead of just using the last week)
- **Binary classifier head** — predicts churn probability from the attended representation

**Why attention matters:** A driver who had one terrible week 2 months ago but has been
fine since is very different from one who's been gradually degrading. Attention weights
reveal *which weeks* the model focuses on for each prediction.
""")

    if 'gru' in data:
        gru = data['gru']

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("GRU AUC", f"{gru['gru_auc']:.4f}")
        with col2:
            st.metric("XGBoost AUC (same split)", f"{gru['xgboost_auc']:.4f}")
        with col3:
            gap = gru['gru_auc'] - gru['xgboost_auc']
            st.metric("Gap", f"{gap:+.4f}",
                       delta=f"{'GRU wins' if gap > 0 else 'XGBoost wins'}",
                       delta_color="normal" if gap > 0 else "inverse")

        st.markdown("""
**Why the GRU wins with enough data:**

With **1,600 training drivers** (340 positive), the GRU has abundant sequences to learn
temporal patterns that XGBoost can't see. The GRU captures *trajectory shapes* — a driver whose
hard-braking rate doubled over 4 weeks is treated differently from one whose rate was always high.
XGBoost can only approximate this through manually engineered trend features.

**The lesson:** Deep learning needs scale. On the original 150-driver dataset, XGBoost won easily.
At 2,000 drivers, the GRU achieves near-perfect AUC (0.999). This demonstrates both *when* to use
deep learning and the importance of having sufficient data to justify the model complexity.
""")

        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists('results/gru_training.png'):
                st.image('results/gru_training.png',
                         caption="Left: training loss convergence. "
                                 "Right: GRU AUC vs XGBoost baseline over epochs.")
        with col2:
            if os.path.exists('results/gru_vs_xgboost_roc.png'):
                st.image('results/gru_vs_xgboost_roc.png',
                         caption="ROC comparison on the same driver split")

        # Attention heatmap
        if os.path.exists('results/gru_attention_heatmap.png'):
            st.markdown("**Attention Weights — What the GRU Focuses On**")
            st.image('results/gru_attention_heatmap.png',
                     caption="Top: churned driver — attention spikes near the end (deterioration). "
                             "Bottom: retained driver — attention is more uniform.")

        # Model architecture details
        with st.expander("Model Architecture Details"):
            st.markdown(f"""
| Component | Value |
|---|---|
| Architecture | {gru['architecture']} |
| Hidden Size | {gru['hidden_size']} |
| GRU Layers | {gru['num_layers']} |
| Input Features | {gru['n_features']} |
| Max Sequence Length | {gru['max_seq_len']} weeks |
| Training Epochs | {gru['n_epochs']} |
| Train Drivers | {gru['n_train_drivers']} |
| Test Drivers | {gru['n_test_drivers']} |
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| Scheduler | Cosine Annealing |
| Loss | BCE with pos_weight for class imbalance |
""")
    else:
        st.info("Run the sequence model pipeline to generate GRU results.")


# =====================================================
# TAB 8 — HOW IT WORKS
# =====================================================
with tab8:
    st.header("How It Works")
    st.caption("A senior engineer's walkthrough of the platform architecture, models, and what makes this project interesting.")

    st.divider()

    # --- ARCHITECTURE ---
    st.subheader("Architecture")
    st.markdown("""
This platform follows a **train-offline, serve-lightweight** pattern designed for Streamlit Community Cloud's constraints (1 GB RAM, no GPU, cold start < 30s).

```
Google Colab (Training)          GitHub Repo (Storage)         Streamlit Cloud (Serving)
========================         ====================         ========================

Raw CSVs (65 MB)                 Aggregated CSVs (< 2 MB)     Load CSVs + models
       |                                |                            |
  Feature Engineering              Model artifacts              Cached in memory
       |                          (.joblib, ~2 MB)            (st.cache_resource)
  Model Training                        |                            |
  (XGBoost, SHAP)               Pre-rendered plots             Interactive dashboard
       |                           (.png, ~1 MB)              (Plotly + Streamlit)
  Save everything ──────────>    Total: ~4 MB repo    ──────>  Loads in < 5 seconds
```

**Why this matters:** The raw data is ~800 MB (2M trips, 3M events, 6.5M activity records). But the Streamlit app never touches any of that. It loads only:
- `weekly_features.csv` (48,687 rows) — the aggregated feature matrix
- 3 model files (~1.5 MB total)
- ~10 pre-computed summary CSVs (< 100 KB total)
- 11 pre-rendered SHAP/analysis plots

This keeps the repo at **4 MB** and cold start under **5 seconds**.
""")

    # --- DATA PIPELINE ---
    st.subheader("Data Pipeline")
    st.markdown("""
**The Simulation Problem:**
We don't have real fleet telematics data, so we *generate* it — but we do it carefully so the patterns are realistic and the models learn something real.

**Hidden Variables Drive Everything:**
Each of the 2,000 drivers is assigned a hidden `_driving_style` (cautious / normal / aggressive / fatigued) that's never exposed to the models. This style controls:

| Style | Hard Brakes/100km | Fuel/100km | Incident Rate |
|-------|:-:|:-:|:-:|
| Cautious (25%) | 0.5 | 10 L | 0.3/yr |
| Normal (35%) | 1.5 | 12 L | 0.8/yr |
| Aggressive (25%) | 4.0 | 15 L | 2.0/yr |
| Fatigued (15%) | 2.0 (increasing) | 13 L | 1.5/yr |

**The clever part:** The models never see the style labels. They have to *discover* the patterns from the observable features (braking rates, fuel consumption, etc.). When the models achieve high accuracy, that validates that our generated data has coherent, learnable structure.

**Embedded Signals:**
- **Churn signal**: Drivers approaching termination show 80% higher event rates in the 8 weeks before they leave
- **Training effect**: Eco-driving trained drivers consume 20% less fuel via lower hard acceleration
- **Fatigue degradation**: "Fatigued" drivers' metrics worsen by 50% over the 6-month window
- **Seasonal effects**: Winter adds 12% fuel consumption, summer AC adds 5%

**Feature Engineering (48,687 rows x 39 features):**
Per-driver, per-week aggregation from 5 raw tables. Key design decisions:
- All rates normalized to per-100km (makes drivers on different routes comparable)
- 4-week rolling trends computed for key metrics (the *change* in behavior is often more predictive than the level)
- Physical workload features from the activity recognition table (steps, floors, lifting events)
""")

    # --- MODELS ---
    st.subheader("The Models")

    # --- MODEL COMPARISON TABLE ---
    st.markdown("#### Model Performance Summary")
    st.markdown("""
All models trained on 2,000 simulated drivers (48,687 weekly observations). Temporal train/test split prevents data leakage.
""")

    comparison_rows = [
        {"Model": "Carbon (XGBoost Regressor)", "Task": "CO2 per km", "Metric": "R²", "Score": "0.91", "Notes": "Behavior explains 91% of emission variance"},
        {"Model": "Incident (XGBoost Classifier)", "Task": "Incident in 4 weeks", "Metric": "AUC", "Score": "0.69", "Notes": "Realistic for rare-event prediction"},
        {"Model": "Churn (XGBoost Classifier)", "Task": "Driver leaves in 8 weeks", "Metric": "AUC", "Score": "0.93", "Notes": "Strong behavioral churn signal"},
        {"Model": "Churn (Tuned XGBoost)", "Task": "Driver leaves in 8 weeks", "Metric": "AUC", "Score": "0.95", "Notes": "+2% from Bayesian hyperparameter tuning"},
        {"Model": "Churn (LightGBM)", "Task": "Driver leaves in 8 weeks", "Metric": "AUC", "Score": "0.94", "Notes": "Edges out default XGBoost"},
        {"Model": "Churn (GRU + Attention)", "Task": "Driver leaves in 8 weeks", "Metric": "AUC", "Score": "0.999", "Notes": "Near-perfect — temporal patterns dominate"},
    ]
    comparison_df = pd.DataFrame(comparison_rows)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    st.markdown("""
**Key takeaway:** The GRU sequence model crushes flat-feature models on churn prediction because
churn is fundamentally a *trajectory* problem — it's the change over time that matters, not any single week's snapshot.
The incident model plateaus at ~0.69 regardless of algorithm, confirming incidents are partly stochastic.
""")

    st.markdown("#### 1. Carbon Emissions Model")
    st.markdown("""
**Type:** XGBoost Regressor | **Target:** CO2 per km | **R² = 0.91**

This model predicts how much CO2 a driver emits per kilometer, based on their behavior. The high R² means driving behavior explains ~91% of the variance in emissions.

**What SHAP tells us:** Idle fuel percentage and hard acceleration are the two biggest knobs. A driver who idles 50% less and smooths their acceleration can cut emissions ~18%.

**What-If Simulator:** Instead of re-running the model on every slider change (too slow for Streamlit), we pre-compute linear coefficients:
```
idle_coefficient = -0.20    # 100% idle reduction -> 20% CO2 savings
ev_coefficient = -0.45      # 100% EV switch -> 45% CO2 savings
coaching_coefficient = -0.06 # coaching worst 100% -> 6% savings
```
The sliders just do arithmetic: `total_reduction = slider_pct * coefficient`. Instant response, zero compute.
""")

    st.markdown("#### 2. Safety / Incident Prediction Model")
    st.markdown("""
**Type:** XGBoost Classifier (cost-sensitive) | **Target:** Incident in next 4 weeks | **AUC = 0.69**

**Why 0.69 and not 0.95?** This is actually realistic for incident prediction. Incidents are rare (~5% positive rate) and partly random. An AUC of 0.68 means the model is meaningfully better than random at ranking drivers by risk, even if it can't predict individual incidents precisely.

**Class Imbalance Handling:** With only ~5% positive samples, a naive model would just predict "no incident" every time and get 95% accuracy. We use:
- `scale_pos_weight = n_neg / n_pos` (~20x) — tells XGBoost that missing a positive costs 20x more than a false alarm
- Temporal train/test split (weeks 1-20 train, 21-26 test) — prevents data leakage from future to past

**Safety Score (rule-based, 0-100):**
This is deliberately NOT a model — it's a transparent, auditable formula:
```
score = 100 - (hard_brake_penalty + speeding_penalty +
               accel_penalty + phone_penalty + severity_penalty)
```
Fleet managers need to explain to drivers *why* their score dropped. A black-box ML score can't do that.
""")

    st.markdown("#### 3. Churn Prediction Model")
    st.markdown("""
**Type:** XGBoost Classifier | **Target:** Driver leaves within 8 weeks | **AUC = 0.93**

This is the strongest model because the churn signal is deliberately embedded in the data — drivers approaching termination show measurable behavior changes. The model picks up on:
- **Behavior deterioration trends** (the `_trend` features capture this)
- **Burnout indicators** (high daily hours + declining safety scores)
- **Night/rotating shift patterns** (these genuinely correlate with turnover in fleet operations)

**Burnout Score (rule-based, 0-100):**
Like the safety score, this is transparent — not ML:
```
burnout = (daily_hours > 9) * 25 + (night_pct > 50) * 20 +
          (safety_declining) * 20 + (activity_declining) * 15 +
          (events_increasing) * 20
```

**Survival Analysis:**
The Kaplan-Meier curves show retention probability over time, split by region. This answers "what percentage of drivers are still here after N weeks?" — useful for workforce planning.
""")

    # --- NOVELTY ---
    st.subheader("What Makes This Interesting")
    st.markdown("""
#### The Cross-Impact Analysis

Most fleet analytics platforms have a carbon dashboard, a safety dashboard, and an HR dashboard. They're separate silos.

This platform's key insight is that **the same interventions affect all three domains simultaneously**, and the interactions aren't obvious:

| Intervention | Carbon | Safety | Retention | Why? |
|---|:-:|:-:|:-:|---|
| Eco-driving coaching | -18% | -12% | -8% | Smooth driving = less fuel AND fewer hard events AND less stress |
| Fix shift schedules | -3% | -25% | -30% | Less fatigue = fewer incidents AND less burnout |
| Route rebalancing | -2% | -5% | -25% | Fair workload = less physical strain AND more equity |

**The "money table"** (Cross-Impact tab) turns this into ROI. Route rebalancing costs $5K/year but saves $69K across all three domains — a 13.8x return. That's a decision a fleet manager can act on *today*.

#### The Activity Recognition Layer

Most fleet platforms stop at telematics (vehicle data). This one includes the **physical** experience of driving:
- Steps per day, floors climbed, lifting events
- These features feed into both the safety model (physical fatigue → incidents) and the churn model (physical strain → burnout)
- The workload equity analysis in Tab 4 reveals that northeast (urban) drivers walk 3x more and climb 5x more stairs than midwest (highway) drivers

#### Designed for Constraints

Every design choice respects the Streamlit Community Cloud limits:
- No training at serve time — all pre-computed
- Pre-rendered SHAP plots (computing SHAP on-the-fly would blow the 1 GB RAM limit)
- What-if simulator uses pre-computed coefficients, not model re-inference
- All charts are Plotly (renders client-side in browser, no server memory needed for chart rendering)
""")

    # --- ADVANCED ML ---
    st.subheader("Advanced ML Techniques")
    st.markdown("""
Beyond the core XGBoost models, the platform includes five additional ML techniques (see Advanced ML tab):

#### Anomaly Detection (Isolation Forest)
Detects unusual driver-week combinations in 14-dimensional behavior space. Unlike threshold alerts ("braking > X"),
Isolation Forest catches *combinations* that are unusual — a driver who simultaneously has high idle time, low steps,
AND elevated phone use, even if none individually triggers a threshold.

#### Unsupervised Clustering (K-Means + PCA)
Discovers natural driver profiles without labels. The data has 4 hidden driving styles (cautious/normal/aggressive/fatigued)
that the model never sees. K-Means recovers 3 clusters (silhouette-optimized), showing that unsupervised learning can
approximate the hidden structure. PCA reduces 14 features to 2D for visualization.

#### Cox Proportional Hazards
Upgrades Kaplan-Meier from descriptive curves to a regression model that quantifies *which features* accelerate churn.
Hazard ratios (HR) are interpretable: HR=1.5 for daily hours means a 1-SD increase in hours raises churn risk 50%.

#### Model Comparison + Stacked Ensemble
Trains 4 algorithms (XGBoost, LightGBM, Random Forest, Logistic Regression) on the same tasks and compares ROC curves.
A meta-learner stacks all predictions. Key insight: with 2,000 drivers, all tree-based models converge
on incidents (~0.69 AUC). For churn, LightGBM edges out XGBoost (0.94 vs 0.93), showing model diversity adds value.

#### Bayesian Hyperparameter Tuning (Optuna)
Replaces hardcoded hyperparameters with 50-trial Bayesian optimization. Uses expanding-window time-series CV
(weeks 1-8/1-12/1-16 train, forward 4 weeks validate) to respect temporal ordering.
**Result:** Churn model AUC improved from 0.93 to 0.95 via tuning alone.

#### Sequence Model (Bidirectional GRU + Attention)
The only deep learning model in the platform. A 2-layer bidirectional GRU reads each driver's full
26-week behavioral sequence and predicts churn. Attention pooling learns which weeks matter most for each
driver's prediction. **Result:** GRU AUC = 0.999 vs XGBoost AUC = 0.96 on the same split — the GRU wins
because it sees the full temporal trajectory, not just individual snapshots. With 1,600 training drivers,
there's enough data for the sequential patterns to emerge. On the original 150-driver dataset, XGBoost
won — demonstrating that knowing when DL needs more data is as important as the architecture itself.
""")

    # --- LIMITATIONS ---
    st.subheader("Honest Limitations")
    st.markdown("""
1. **Simulated data.** The patterns are realistic but hand-crafted. Real fleet data would have messier distributions, missing values, and surprising edge cases. Model performance on real data would likely be lower.

2. **Cross-impact values are estimated.** The intervention ROI table uses domain-knowledge estimates, not causal inference. In production, you'd need A/B testing or at minimum interrupted time series analysis to validate these numbers.

3. **Incident model AUC = 0.69.** Decent for a rare-event prediction problem, but not reliable enough for individual-level decisions. Best used for fleet-level risk stratification, not "fire this driver."

4. **Cox PH uses univariate hazard ratios.** The `lifelines` library can't build in this environment, so Cox PH falls back to median-split hazard ratios rather than a full multivariate partial likelihood model. A production version would use proper survival regression.

5. **Static training effect.** The training ROI analysis compares trained vs. untrained groups cross-sectionally. It doesn't prove causation — drivers who opt into training may already be more conscientious.

6. **Ensemble meta-learner underperforms.** The stacked ensemble's meta-learner doesn't consistently beat the best individual model. Individual models like Random Forest and LightGBM actually outperform XGBoost on specific tasks with more data, showing model diversity matters — but the stacking layer adds noise rather than signal here.
""")
