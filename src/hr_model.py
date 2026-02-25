"""
HR / Churn Prediction Model
Workforce retention prediction, burnout detection, and training ROI.
Uses XGBoost for churn prediction and Kaplan-Meier for survival analysis.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)


def create_churn_labels(features_df, drivers_df):
    """Create churn labels: does this driver leave within 8 weeks?"""
    df = features_df.copy()
    df['churns_8weeks'] = 0

    for driver_id in df['driver_id'].unique():
        driver = drivers_df[drivers_df['driver_id'] == driver_id].iloc[0]
        if pd.isna(driver['termination_date']):
            continue
        term_week = pd.to_datetime(driver['termination_date']).isocalendar().week
        term_week = int(term_week)
        mask = df['driver_id'] == driver_id
        df.loc[mask, 'churns_8weeks'] = df.loc[mask, 'week'].apply(
            lambda w: 1 if 0 < (term_week - w) <= 8 else 0
        )

    return df


def compute_burnout_scores(features_df):
    """Rule-based burnout risk score 0-100."""
    df = features_df.copy()

    score = np.zeros(len(df))
    score += (df['daily_hours'] > 9).astype(int) * 25
    score += (df['night_driving_pct'] > 50).astype(int) * 20

    if 'safety_score_trend' in df.columns:
        score += (df['safety_score_trend'].fillna(0) < -5).astype(int) * 20

    score += (df['daily_steps_trend'].fillna(0) < -500).astype(int) * 15
    score += (df['total_events_per_100km_trend'].fillna(0) > 0.5).astype(int) * 20

    df['burnout_score'] = score.clip(0, 100)
    return df


def train_churn_model(features_df, drivers_df, safety_scores_df):
    """Train XGBoost to predict churn within 8 weeks."""
    # Merge safety scores
    df = features_df.merge(safety_scores_df, on=['driver_id', 'week'], how='left')

    df['safety_score_trend'] = df.groupby('driver_id')['safety_score'].transform(
        lambda x: x.diff(4)
    )

    # Add churn labels
    df = create_churn_labels(df, drivers_df)

    # Add burnout scores
    df = compute_burnout_scores(df)

    feature_cols = [
        'hard_brakes_per_100km', 'hard_accels_per_100km',
        'speeding_per_100km', 'total_events_per_100km',
        'idle_time_pct', 'daily_hours', 'daily_km',
        'trips_per_day', 'night_driving_pct',
        'daily_steps', 'daily_floors', 'daily_lifting_events',
        'avg_package_weight_kg', 'daily_walk_minutes',
        'age', 'experience_years',
        'has_eco_training', 'has_safety_training',
        'safety_score', 'safety_score_trend',
        'burnout_score',
        'total_events_per_100km_trend', 'fuel_per_100km_trend',
        'daily_hours_trend', 'daily_steps_trend', 'co2_per_km_trend',
    ]

    # Temporal split
    train_mask = df['week'] <= 20
    test_mask = df['week'] > 20

    X_train = df.loc[train_mask, feature_cols].fillna(0)
    y_train = df.loc[train_mask, 'churns_8weeks']
    X_test = df.loc[test_mask, feature_cols].fillna(0)
    y_test = df.loc[test_mask, 'churns_8weeks']

    n_pos = max((y_train == 1).sum(), 1)
    n_neg = (y_train == 0).sum()
    scale_weight = n_neg / n_pos

    print(f"Train: {len(y_train)} samples, {n_pos} positive ({n_pos/len(y_train)*100:.1f}%)")
    print(f"Test:  {len(y_test)} samples, {(y_test==1).sum()} positive")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]

    if y_test.nunique() > 1:
        auc = roc_auc_score(y_test, y_prob)
        print(f"Churn Model AUC-ROC: {auc:.4f}")
    else:
        print("Warning: only one class in test set")

    joblib.dump(model, 'models/churn_model.joblib')

    # Update feature columns
    fc_path = 'models/feature_columns.json'
    if os.path.exists(fc_path):
        with open(fc_path) as f:
            fc = json.load(f)
    else:
        fc = {}
    fc['churn'] = feature_cols
    with open(fc_path, 'w') as f:
        json.dump(fc, f)

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    np.save('models/churn_shap_values.npy', shap_values)
    np.save('models/churn_shap_X_test.npy', X_test.values)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig('results/churn_shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Generate churn predictions for all drivers (latest week)
    latest = df.groupby('driver_id').tail(1).copy()
    X_latest = latest[feature_cols].fillna(0)
    latest['churn_prob'] = model.predict_proba(X_latest)[:, 1]

    predictions = latest[['driver_id', 'week', 'burnout_score']].copy()
    predictions['churn_prob'] = latest['churn_prob'].values

    # Top factors via SHAP
    shap_latest = explainer.shap_values(X_latest)
    top_factors_list = []
    for i in range(len(X_latest)):
        abs_shap = np.abs(shap_latest[i])
        top_idx = np.argsort(abs_shap)[-3:][::-1]
        top_factors_list.append({
            'top_factor_1': feature_cols[top_idx[0]],
            'top_factor_2': feature_cols[top_idx[1]],
            'top_factor_3': feature_cols[top_idx[2]],
        })

    top_df = pd.DataFrame(top_factors_list)
    predictions = predictions.reset_index(drop=True)
    predictions = pd.concat([predictions, top_df], axis=1)
    predictions = predictions.sort_values('churn_prob', ascending=False)
    predictions.to_csv('data/churn_predictions.csv', index=False)
    print(f"Churn predictions saved: {len(predictions)} drivers")

    # Burnout scatter plot
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(
        latest['daily_hours'],
        latest['safety_score'] if 'safety_score' in latest.columns else latest['total_events_per_100km'],
        c=latest['burnout_score'],
        s=latest['churn_prob'].values * 200 + 20,
        cmap='RdYlGn_r',
        alpha=0.7,
        edgecolors='gray'
    )
    plt.colorbar(sc, label='Burnout Score')
    ax.set_xlabel('Daily Hours')
    ax.set_ylabel('Safety Score')
    ax.set_title('Driver Burnout Risk\n(size = churn probability, color = burnout score)')
    plt.tight_layout()
    plt.savefig('results/burnout_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()

    return model, feature_cols, df


def compute_survival_curves(drivers_df, features_df):
    """Compute Kaplan-Meier survival curves using numpy (no lifelines)."""
    # Compute tenure in weeks for each driver
    tenure_data = []
    for _, driver in drivers_df.iterrows():
        hire_date = pd.to_datetime(driver['hire_date'])
        if pd.notna(driver['termination_date']):
            end_date = pd.to_datetime(driver['termination_date'])
            event = 1  # churned
        else:
            end_date = pd.to_datetime('2024-07-01')
            event = 0  # censored
        tenure_weeks = max(1, (end_date - hire_date).days // 7)
        tenure_data.append({
            'driver_id': driver['driver_id'],
            'tenure_weeks': tenure_weeks,
            'event': event,
            'region': driver['region'],
        })

    tenure_df = pd.DataFrame(tenure_data)

    # Kaplan-Meier curve
    fig, ax = plt.subplots(figsize=(10, 6))

    for region in ['northeast', 'southeast', 'midwest']:
        subset = tenure_df[tenure_df['region'] == region]
        times = sorted(subset['tenure_weeks'].unique())
        n_at_risk = len(subset)
        survival = []
        prob = 1.0

        for t in times:
            events = ((subset['tenure_weeks'] == t) & (subset['event'] == 1)).sum()
            censored = ((subset['tenure_weeks'] == t) & (subset['event'] == 0)).sum()
            if n_at_risk > 0:
                prob *= (1 - events / n_at_risk)
            n_at_risk -= (events + censored)
            survival.append(prob)

        ax.step(times, survival, where='post', label=region.capitalize(), linewidth=2)

    ax.set_xlabel('Tenure (weeks)')
    ax.set_ylabel('Survival Probability')
    ax.set_title('Driver Retention - Kaplan-Meier Survival Curves')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/survival_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save survival model data
    joblib.dump(tenure_df, 'models/survival_model.pkl')
    print("Survival curves generated")

    return tenure_df


def compute_training_roi(features_df, drivers_df):
    """Compute training ROI: before/after metrics per training type."""
    df = features_df.copy()

    training_types = {
        'eco_driving': {
            'metric': 'fuel_per_100km',
            'metric_name': 'Fuel per 100km (L)',
            'cost_per_driver': 200,
        },
        'defensive_driving': {
            'metric': 'hard_brakes_per_100km',
            'metric_name': 'Hard Brakes per 100km',
            'cost_per_driver': 200,
        },
    }

    results = []
    for training_type, config in training_types.items():
        col = f'has_{training_type.replace("_driving", "")}_training' if 'eco' in training_type else 'has_safety_training'
        if 'eco' in training_type:
            col = 'has_eco_training'
        else:
            col = 'has_safety_training'

        trained = df[df[col] == True]
        untrained = df[df[col] == False]

        trained_mean = trained[config['metric']].mean()
        untrained_mean = untrained[config['metric']].mean()
        improvement_pct = (untrained_mean - trained_mean) / untrained_mean * 100

        n_trained = trained['driver_id'].nunique()
        total_cost = n_trained * config['cost_per_driver']

        # Estimate savings
        if 'fuel' in config['metric']:
            # Fuel savings in liters * ~$1.5/L
            fuel_saved_per_driver = (untrained_mean - trained_mean) / 100 * df['total_km'].mean()
            annual_savings = fuel_saved_per_driver * 1.5 * n_trained * 2  # annualize
        else:
            # Fewer incidents = fewer claims
            annual_savings = improvement_pct / 100 * 50000  # rough estimate

        results.append({
            'training_type': training_type,
            'metric': config['metric_name'],
            'trained_avg': round(trained_mean, 2),
            'untrained_avg': round(untrained_mean, 2),
            'improvement_pct': round(improvement_pct, 1),
            'n_trained_drivers': n_trained,
            'total_cost': total_cost,
            'estimated_annual_savings': round(annual_savings, 0),
            'roi': round(annual_savings / max(total_cost, 1), 1),
        })

    roi_df = pd.DataFrame(results)
    roi_df.to_csv('data/training_roi.csv', index=False)
    print(f"Training ROI:\n{roi_df}")

    # Training impact plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, row in roi_df.iterrows():
        ax = axes[i]
        bars = ax.bar(
            ['Untrained', 'Trained'],
            [row['untrained_avg'], row['trained_avg']],
            color=['#e74c3c', '#2ecc71'],
            edgecolor='black'
        )
        ax.set_title(f'{row["training_type"].replace("_", " ").title()}\n{row["metric"]}')
        ax.set_ylabel(row['metric'])
        pct = row['improvement_pct']
        ax.annotate(f'{pct:+.1f}%', xy=(1, row['trained_avg']),
                   fontsize=14, fontweight='bold', color='green',
                   ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('results/training_impact.png', dpi=150, bbox_inches='tight')
    plt.close()

    return roi_df


if __name__ == '__main__':
    print("Loading data...")
    features_df = pd.read_csv('data/weekly_features.csv')
    drivers_df = pd.read_csv('data/drivers.csv')
    safety_scores = pd.read_csv('data/driver_safety_scores.csv')

    print("\n--- Training Churn Model ---")
    model, feature_cols, enriched_df = train_churn_model(
        features_df, drivers_df, safety_scores
    )

    print("\n--- Computing Survival Curves ---")
    compute_survival_curves(drivers_df, features_df)

    print("\n--- Computing Training ROI ---")
    compute_training_roi(features_df, drivers_df)

    print("\nHR module complete.")
