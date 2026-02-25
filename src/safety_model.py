"""
Safety / Incident Prediction Model
Builds safety scores and predicts incident risk.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)


def compute_safety_scores(features_df):
    """
    Composite safety score 0-100 per driver per week.
    Higher = safer.
    """
    df = features_df.copy()

    # Compute fleet 95th percentiles for normalization
    p95 = {
        'hard_brakes': df['hard_brakes_per_100km'].quantile(0.95),
        'speeding': df['speeding_per_100km'].quantile(0.95),
        'hard_accels': df['hard_accels_per_100km'].quantile(0.95),
        'phone_use': df['phone_use_per_100km'].quantile(0.95),
    }

    # Avoid division by zero
    for k in p95:
        if p95[k] == 0:
            p95[k] = 1

    scores = 100 - (
        (df['hard_brakes_per_100km'] / p95['hard_brakes']).clip(0, 1) * 25 +
        (df['speeding_per_100km'] / p95['speeding']).clip(0, 1) * 25 +
        (df['hard_accels_per_100km'] / p95['hard_accels']).clip(0, 1) * 15 +
        (df['phone_use_per_100km'] / p95['phone_use']).clip(0, 1) * 20 +
        df['high_severity_events'].clip(0, 3) * 5
    )

    scores = scores.clip(0, 100)

    result = df[['driver_id', 'week']].copy()
    result['safety_score'] = scores.round(1)

    result.to_csv('data/driver_safety_scores.csv', index=False)
    print(f"Safety scores: {len(result)} rows")
    print(f"Mean safety score: {result['safety_score'].mean():.1f}")
    print(f"Std: {result['safety_score'].std():.1f}")

    # Distribution plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(result['safety_score'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_xlabel('Safety Score')
    ax.set_ylabel('Count')
    ax.set_title('Fleet Safety Score Distribution')
    ax.axvline(result['safety_score'].mean(), color='red', linestyle='--',
               label=f'Mean: {result["safety_score"].mean():.1f}')
    ax.legend()
    plt.tight_layout()
    plt.savefig('results/safety_score_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    return result


def train_incident_model(features_df, incidents_df, safety_scores_df):
    """
    Predict whether a driver has an incident in the next 4 weeks.
    Uses temporal train/test split and cost-sensitive XGBoost.
    """
    # Merge safety scores
    df = features_df.merge(safety_scores_df, on=['driver_id', 'week'], how='left')

    # Create safety_score_trend
    df['safety_score_trend'] = df.groupby('driver_id')['safety_score'].transform(
        lambda x: x.diff(4)
    )

    # Create incident labels: did this driver have an incident in weeks [w+1, w+4]?
    incident_weeks = {}
    for _, row in incidents_df.iterrows():
        did = row['driver_id']
        inc_week = pd.to_datetime(row['date']).isocalendar().week
        if did not in incident_weeks:
            incident_weeks[did] = set()
        incident_weeks[did].add(int(inc_week))

    df['incident_next_4weeks'] = 0
    for driver_id in df['driver_id'].unique():
        iw = incident_weeks.get(driver_id, set())
        if not iw:
            continue
        mask = df['driver_id'] == driver_id
        df.loc[mask, 'incident_next_4weeks'] = df.loc[mask, 'week'].apply(
            lambda w: 1 if any(i in range(w + 1, w + 5) for i in iw) else 0
        )

    feature_cols = [
        'hard_brakes_per_100km', 'hard_accels_per_100km',
        'speeding_per_100km', 'sharp_turns_per_100km',
        'phone_use_per_100km', 'high_severity_events',
        'total_events_per_100km', 'idle_time_pct',
        'daily_hours', 'daily_km', 'trips_per_day',
        'night_driving_pct', 'daily_steps', 'daily_floors',
        'daily_lifting_events', 'avg_package_weight_kg',
        'age', 'experience_years',
        'has_eco_training', 'has_safety_training',
        'safety_score', 'safety_score_trend',
        'total_events_per_100km_trend', 'fuel_per_100km_trend',
        'daily_hours_trend', 'daily_steps_trend',
    ]

    # Temporal split: weeks 1-20 train, weeks 21-26 test
    train_mask = df['week'] <= 20
    test_mask = df['week'] > 20

    X_train = df.loc[train_mask, feature_cols].fillna(0)
    y_train = df.loc[train_mask, 'incident_next_4weeks']
    X_test = df.loc[test_mask, feature_cols].fillna(0)
    y_test = df.loc[test_mask, 'incident_next_4weeks']

    # Handle class imbalance
    n_neg = (y_train == 0).sum()
    n_pos = max((y_train == 1).sum(), 1)
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
        print(f"Incident Model AUC-ROC: {auc:.4f}")

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Incident Prediction - ROC Curve')
        ax.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig('results/incident_roc.png', dpi=150, bbox_inches='tight')
        plt.close()
    else:
        print("Warning: only one class in test set, skipping AUC")

    # Save model
    joblib.dump(model, 'models/incident_model.joblib')

    # Update feature columns JSON
    fc_path = 'models/feature_columns.json'
    if os.path.exists(fc_path):
        with open(fc_path) as f:
            fc = json.load(f)
    else:
        fc = {}
    fc['incident'] = feature_cols
    with open(fc_path, 'w') as f:
        json.dump(fc, f)

    # SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    np.save('models/incident_shap_values.npy', shap_values)
    np.save('models/incident_shap_X_test.npy', X_test.values)

    # SHAP summary plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig('results/safety_shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Generate incident predictions for all drivers (latest week)
    latest = df.groupby('driver_id').tail(1).copy()
    X_latest = latest[feature_cols].fillna(0)
    latest['risk_score'] = model.predict_proba(X_latest)[:, 1]

    # Get top 3 factors per driver using SHAP
    explainer_latest = shap.TreeExplainer(model)
    shap_latest = explainer_latest.shap_values(X_latest)

    top_factors = []
    for i in range(len(X_latest)):
        abs_shap = np.abs(shap_latest[i])
        top_idx = np.argsort(abs_shap)[-3:][::-1]
        top_factors.append({
            'top_factor_1': feature_cols[top_idx[0]],
            'top_factor_2': feature_cols[top_idx[1]],
            'top_factor_3': feature_cols[top_idx[2]],
        })

    top_factors_df = pd.DataFrame(top_factors)
    predictions = latest[['driver_id', 'week']].reset_index(drop=True)
    predictions['risk_score'] = latest['risk_score'].values
    predictions = pd.concat([predictions.reset_index(drop=True),
                            top_factors_df.reset_index(drop=True)], axis=1)
    predictions = predictions.sort_values('risk_score', ascending=False)
    predictions.to_csv('data/incident_predictions.csv', index=False)
    print(f"Incident predictions saved: {len(predictions)} drivers")

    return model, feature_cols


if __name__ == '__main__':
    print("Loading data...")
    features_df = pd.read_csv('data/weekly_features.csv')
    incidents_df = pd.read_csv('data/incidents.csv')

    print("\n--- Computing Safety Scores ---")
    safety_scores = compute_safety_scores(features_df)

    print("\n--- Training Incident Model ---")
    train_incident_model(features_df, incidents_df, safety_scores)

    print("\nSafety module complete.")
