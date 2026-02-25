"""
Carbon / Emissions Model
Predicts CO2 per km from driver behavior, computes reduction opportunities,
and builds what-if simulator coefficients.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)


def train_carbon_model(features_df):
    """Train XGBoost to predict co2_per_km from driving behavior."""

    feature_cols = [
        'hard_brakes_per_100km', 'hard_accels_per_100km',
        'speeding_per_100km', 'sharp_turns_per_100km',
        'phone_use_per_100km', 'idle_time_pct', 'idle_fuel_pct',
        'daily_hours', 'daily_km', 'trips_per_day',
        'night_driving_pct', 'age', 'experience_years',
        'has_eco_training', 'has_safety_training',
    ]

    # Encode categoricals
    df = features_df.copy()
    df['region_northeast'] = (df['region'] == 'northeast').astype(int)
    df['region_southeast'] = (df['region'] == 'southeast').astype(int)
    df['region_midwest'] = (df['region'] == 'midwest').astype(int)
    df['shift_day'] = (df['shift'] == 'day').astype(int)
    df['shift_night'] = (df['shift'] == 'night').astype(int)
    df['shift_rotating'] = (df['shift'] == 'rotating').astype(int)

    feature_cols += [
        'region_northeast', 'region_southeast', 'region_midwest',
        'shift_day', 'shift_night', 'shift_rotating',
    ]

    # Filter valid rows
    df = df.dropna(subset=['co2_per_km'])
    df = df[df['co2_per_km'] > 0]

    X = df[feature_cols].fillna(0)
    y = df['co2_per_km']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Carbon Model - MAE: {mae:.4f}, R²: {r2:.4f}")

    # Save model
    joblib.dump(model, 'models/carbon_model.joblib')

    # Save feature columns
    with open('models/feature_columns.json', 'w') as f:
        json.dump({'carbon': feature_cols}, f)

    # SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    np.save('models/carbon_shap_values.npy', shap_values)
    np.save('models/carbon_shap_X_test.npy', X_test.values)

    # SHAP summary plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig('results/carbon_shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Carbon by behavior plot
    fig, ax = plt.subplots(figsize=(10, 6))
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    importances.sort_values(ascending=True).tail(10).plot(kind='barh', ax=ax)
    ax.set_title('Top 10 Features Driving CO₂ Emissions')
    ax.set_xlabel('Feature Importance')
    plt.tight_layout()
    plt.savefig('results/carbon_by_behavior.png', dpi=150, bbox_inches='tight')
    plt.close()

    return model, feature_cols


def compute_driver_carbon_summary(features_df):
    """Per-driver CO2 summary and gap analysis."""
    summary = features_df.groupby('driver_id').agg({
        'co2_kg': 'sum',
        'co2_per_km': 'mean',
        'total_km': 'sum',
        'fuel_per_100km': 'mean',
        'idle_time_pct': 'mean',
        'idle_fuel_pct': 'mean',
        'hard_accels_per_100km': 'mean',
        'speeding_per_100km': 'mean',
        'region': 'first',
        'has_eco_training': 'first',
    }).reset_index()

    # Compute gap vs median
    median_co2_per_km = summary['co2_per_km'].median()
    summary['co2_gap_per_km'] = summary['co2_per_km'] - median_co2_per_km
    summary['reducible_co2_kg'] = np.maximum(0, summary['co2_gap_per_km'] * summary['total_km'])
    summary = summary.sort_values('reducible_co2_kg', ascending=False)

    summary.to_csv('data/driver_carbon_summary.csv', index=False)
    print(f"Driver carbon summary: {len(summary)} drivers")
    print(f"Fleet median CO2/km: {median_co2_per_km:.4f}")
    print(f"Total reducible CO2: {summary['reducible_co2_kg'].sum():.0f} kg")

    return summary


def compute_fleet_carbon_monthly(features_df):
    """Monthly fleet CO2 totals for ESG reporting."""
    monthly = features_df.groupby('month').agg({
        'co2_kg': 'sum',
        'total_km': 'sum',
        'fuel_per_100km': 'mean',
        'idle_fuel_pct': 'mean',
    }).reset_index()
    monthly['co2_per_km'] = monthly['co2_kg'] / monthly['total_km']
    monthly['month_name'] = monthly['month'].map(
        {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun'}
    )
    monthly.to_csv('data/fleet_carbon_monthly.csv', index=False)
    print(f"Fleet carbon monthly: {len(monthly)} months")
    return monthly


def compute_whatif_coefficients(features_df):
    """Pre-compute coefficients for the what-if simulator."""
    total_co2 = features_df['co2_kg'].sum()
    total_idle_co2 = (features_df['idle_fuel_pct'] / 100 * features_df['co2_kg']).sum()
    idle_share = total_idle_co2 / total_co2

    # Worst 20% drivers' excess CO2
    median_co2 = features_df['co2_per_km'].median()
    worst_20 = features_df[features_df['co2_per_km'] > features_df['co2_per_km'].quantile(0.8)]
    excess_co2 = ((worst_20['co2_per_km'] - median_co2) * worst_20['total_km']).sum()
    coaching_share = excess_co2 / total_co2

    # Diesel share of emissions
    diesel_mask = features_df['fuel_per_100km'] > 0
    diesel_co2 = features_df.loc[diesel_mask, 'co2_kg'].sum()
    diesel_share = diesel_co2 / total_co2

    # EV replacement saves ~45% of diesel CO2 (accounting for grid factor)
    ev_savings_factor = 0.45

    coefficients = {
        'idle_coefficient': round(-idle_share, 4),
        'ev_coefficient': round(-diesel_share * ev_savings_factor, 4),
        'coaching_coefficient': round(-coaching_share, 4),
        'total_co2_kg': round(total_co2, 0),
        'total_co2_tons': round(total_co2 / 1000, 1),
    }

    with open('data/whatif_coefficients.json', 'w') as f:
        json.dump(coefficients, f, indent=2)

    print(f"What-if coefficients: {coefficients}")
    return coefficients


if __name__ == '__main__':
    print("Loading features...")
    features_df = pd.read_csv('data/weekly_features.csv')

    print("\n--- Training Carbon Model ---")
    model, feature_cols = train_carbon_model(features_df)

    print("\n--- Computing Driver Carbon Summary ---")
    compute_driver_carbon_summary(features_df)

    print("\n--- Computing Fleet Carbon Monthly ---")
    compute_fleet_carbon_monthly(features_df)

    print("\n--- Computing What-If Coefficients ---")
    compute_whatif_coefficients(features_df)

    print("\nCarbon module complete.")
