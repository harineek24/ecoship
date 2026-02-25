"""
Advanced ML Models
Anomaly Detection, Driver Clustering, Cox PH Survival Regression,
Stacked Ensemble with Model Comparison, and Hyperparameter Tuning.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, roc_auc_score, roc_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)


# =========================================================
# 1. ANOMALY DETECTION — Isolation Forest
# =========================================================

def train_anomaly_detector(features_df):
    """
    Isolation Forest to detect unusual driver-week behavior.
    Flags weeks where a driver's pattern deviates significantly
    from the fleet norm — catches things a threshold never would.
    """
    behavior_cols = [
        'hard_brakes_per_100km', 'hard_accels_per_100km',
        'speeding_per_100km', 'sharp_turns_per_100km',
        'phone_use_per_100km', 'total_events_per_100km',
        'fuel_per_100km', 'idle_time_pct',
        'daily_hours', 'daily_km', 'night_driving_pct',
        'daily_steps', 'daily_floors', 'daily_lifting_events',
    ]

    df = features_df.copy()
    X = df[behavior_cols].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(X_scaled)

    df['anomaly_score'] = iso_forest.score_samples(X_scaled)
    df['is_anomaly'] = iso_forest.predict(X_scaled) == -1

    anomalies = df[['driver_id', 'week', 'anomaly_score', 'is_anomaly']].copy()
    anomalies.to_csv('data/anomaly_scores.csv', index=False)

    joblib.dump(iso_forest, 'models/isolation_forest.joblib')
    joblib.dump(scaler, 'models/anomaly_scaler.joblib')

    # Find what makes each anomaly anomalous (z-score analysis)
    anomalous_weeks = df[df['is_anomaly']].copy()
    fleet_means = X.mean()
    fleet_stds = X.std()

    anomaly_reasons = []
    for _, row in anomalous_weeks.iterrows():
        z_scores = {}
        for col in behavior_cols:
            z = (row[col] - fleet_means[col]) / max(fleet_stds[col], 1e-9)
            z_scores[col] = abs(z)
        top_3 = sorted(z_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        anomaly_reasons.append({
            'driver_id': row['driver_id'],
            'week': row['week'],
            'anomaly_score': round(row['anomaly_score'], 4),
            'reason_1': top_3[0][0],
            'reason_1_zscore': round(top_3[0][1], 2),
            'reason_2': top_3[1][0],
            'reason_2_zscore': round(top_3[1][1], 2),
            'reason_3': top_3[2][0],
            'reason_3_zscore': round(top_3[2][1], 2),
        })

    reasons_df = pd.DataFrame(anomaly_reasons)
    reasons_df.to_csv('data/anomaly_details.csv', index=False)

    # Plot: anomaly score distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['anomaly_score'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    threshold = df[df['is_anomaly']]['anomaly_score'].max()
    ax.axvline(threshold, color='red', linestyle='--',
               label=f'Anomaly threshold: {threshold:.3f}')
    ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('Count')
    ax.set_title('Isolation Forest Anomaly Score Distribution')
    ax.legend()
    plt.tight_layout()
    plt.savefig('results/anomaly_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    n_anomalies = df['is_anomaly'].sum()
    n_drivers = anomalous_weeks['driver_id'].nunique()
    print(f"Anomaly Detection: {n_anomalies} anomalous weeks "
          f"({n_anomalies/len(df)*100:.1f}%)")
    print(f"Anomalous drivers: {n_drivers}")

    return iso_forest, anomalies


# =========================================================
# 2. DRIVER CLUSTERING — K-Means + PCA
# =========================================================

def cluster_drivers(features_df):
    """
    K-Means clustering to discover driver behavior profiles.
    Uses aggregated per-driver features with PCA for visualization.
    The data has hidden driving styles — can the model recover them?
    """
    cluster_cols = [
        'hard_brakes_per_100km', 'hard_accels_per_100km',
        'speeding_per_100km', 'phone_use_per_100km',
        'total_events_per_100km', 'fuel_per_100km',
        'idle_time_pct', 'daily_hours', 'daily_km',
        'night_driving_pct', 'daily_steps', 'daily_floors',
        'daily_lifting_events', 'co2_per_km',
    ]

    driver_agg = features_df.groupby('driver_id')[cluster_cols].mean().reset_index()
    X = driver_agg[cluster_cols].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Find optimal k using silhouette score
    silhouette_scores = {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        silhouette_scores[k] = round(sil, 4)

    optimal_k = max(silhouette_scores, key=silhouette_scores.get)
    print(f"Optimal k: {optimal_k} (silhouette: {silhouette_scores[optimal_k]:.3f})")

    final_km = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    driver_agg['cluster'] = final_km.fit_predict(X_scaled)

    # PCA for 2D visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    driver_agg['pca_1'] = X_pca[:, 0]
    driver_agg['pca_2'] = X_pca[:, 1]

    driver_agg.to_csv('data/driver_clusters.csv', index=False)
    joblib.dump(final_km, 'models/kmeans_model.joblib')
    joblib.dump(scaler, 'models/cluster_scaler.joblib')
    joblib.dump(pca, 'models/cluster_pca.joblib')

    # Cluster profiles (mean of each feature per cluster)
    profiles = driver_agg.groupby('cluster')[cluster_cols].mean().round(2)
    profiles.to_csv('data/cluster_profiles.csv')

    # Silhouette score plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = list(silhouette_scores.keys())
    scores = list(silhouette_scores.values())
    ax.plot(ks, scores, 'bo-', linewidth=2, markersize=8)
    ax.axvline(optimal_k, color='red', linestyle='--', alpha=0.7,
               label=f'Optimal k={optimal_k}')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('K-Means: Optimal Cluster Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/silhouette_scores.png', dpi=150, bbox_inches='tight')
    plt.close()

    # PCA scatter colored by cluster
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=driver_agg['cluster'], cmap='Set2',
        s=60, alpha=0.8, edgecolors='gray'
    )
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    ax.set_title('Driver Clusters (PCA Projection)')
    plt.colorbar(scatter, label='Cluster', ticks=range(optimal_k))
    plt.tight_layout()
    plt.savefig('results/cluster_pca.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save metadata
    sil_df = pd.DataFrame({'k': ks, 'silhouette_score': scores})
    sil_df.to_csv('data/silhouette_scores.csv', index=False)

    pca_info = {
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'optimal_k': optimal_k,
        'silhouette_score': silhouette_scores[optimal_k],
    }
    with open('data/cluster_info.json', 'w') as f:
        json.dump(pca_info, f, indent=2)

    for c in range(optimal_k):
        n = (driver_agg['cluster'] == c).sum()
        print(f"  Cluster {c}: {n} drivers")

    return final_km, driver_agg


# =========================================================
# 3. COX PROPORTIONAL HAZARDS — Survival Regression
# =========================================================

def train_cox_model(drivers_df, features_df, safety_scores_df):
    """
    Cox PH survival regression: which features accelerate churn?
    Returns hazard ratios with confidence intervals.
    """
    # Aggregate features to per-driver
    driver_features = features_df.groupby('driver_id').agg({
        'daily_hours': 'mean',
        'night_driving_pct': 'mean',
        'total_events_per_100km': 'mean',
        'daily_steps': 'mean',
        'daily_lifting_events': 'mean',
        'fuel_per_100km': 'mean',
        'hard_brakes_per_100km': 'mean',
    }).reset_index()

    safety_avg = safety_scores_df.groupby('driver_id')['safety_score'].mean().reset_index()
    driver_features = driver_features.merge(safety_avg, on='driver_id', how='left')

    # Build survival dataset
    tenure_data = []
    for _, driver in drivers_df.iterrows():
        hire_date = pd.to_datetime(driver['hire_date'])
        if pd.notna(driver['termination_date']):
            end_date = pd.to_datetime(driver['termination_date'])
            event = 1
        else:
            end_date = pd.to_datetime('2024-07-01')
            event = 0
        tenure_weeks = max(1, (end_date - hire_date).days // 7)
        tenure_data.append({
            'driver_id': driver['driver_id'],
            'tenure_weeks': tenure_weeks,
            'event': event,
            'age': driver['age'],
            'experience_years': driver['experience_years'],
        })

    tenure_df = pd.DataFrame(tenure_data)
    survival_df = tenure_df.merge(driver_features, on='driver_id', how='left')
    survival_df = survival_df.dropna()

    covariate_cols = [
        'age', 'experience_years', 'daily_hours', 'night_driving_pct',
        'total_events_per_100km', 'daily_steps', 'daily_lifting_events',
        'safety_score', 'hard_brakes_per_100km',
    ]

    try:
        from lifelines import CoxPHFitter

        cox_df = survival_df[['tenure_weeks', 'event'] + covariate_cols].copy()

        # Standardize covariates for comparable hazard ratios
        scaler = StandardScaler()
        cox_df[covariate_cols] = scaler.fit_transform(cox_df[covariate_cols])

        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(cox_df, duration_col='tenure_weeks', event_col='event')

        summary = cph.summary
        summary.to_csv('data/cox_summary.csv')

        hazard_ratios = pd.DataFrame({
            'feature': summary.index,
            'hazard_ratio': np.exp(summary['coef']).values,
            'coef': summary['coef'].values,
            'se': summary['se(coef)'].values,
            'p_value': summary['p'].values,
            'lower_ci': np.exp(
                summary['coef'].values - 1.96 * summary['se(coef)'].values
            ),
            'upper_ci': np.exp(
                summary['coef'].values + 1.96 * summary['se(coef)'].values
            ),
        })
        hazard_ratios.to_csv('data/cox_hazard_ratios.csv', index=False)

        concordance = cph.concordance_index_
        joblib.dump(cph, 'models/cox_model.joblib')

        print(f"Cox PH Model — Concordance Index: {concordance:.3f}")
        for _, row in hazard_ratios.sort_values('hazard_ratio', ascending=False).iterrows():
            sig = ('***' if row['p_value'] < 0.001 else
                   '**' if row['p_value'] < 0.01 else
                   '*' if row['p_value'] < 0.05 else '')
            print(f"  {row['feature']:30s} HR={row['hazard_ratio']:.3f} "
                  f"(p={row['p_value']:.4f}) {sig}")

        # Forest plot of hazard ratios
        fig, ax = plt.subplots(figsize=(10, 8))
        hr_sorted = hazard_ratios.sort_values('hazard_ratio')
        y_pos = range(len(hr_sorted))

        colors = ['#e74c3c' if hr > 1 else '#27ae60'
                  for hr in hr_sorted['hazard_ratio']]
        ax.barh(y_pos, hr_sorted['hazard_ratio'] - 1, left=1,
                color=colors, alpha=0.7, edgecolor='black')

        for i, (_, row) in enumerate(hr_sorted.iterrows()):
            ax.plot([row['lower_ci'], row['upper_ci']], [i, i],
                    'k-', linewidth=1.5)

        ax.axvline(1, color='black', linewidth=1, linestyle='-')
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(
            [f.replace('_', ' ') for f in hr_sorted['feature']]
        )
        ax.set_xlabel('Hazard Ratio (per 1 SD increase)')
        ax.set_title('Cox PH: What Accelerates Driver Churn?\n'
                      '(HR > 1 = increases risk, HR < 1 = protective)')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/cox_hazard_ratios.png', dpi=150, bbox_inches='tight')
        plt.close()

        cox_info = {
            'concordance_index': round(concordance, 4),
            'n_subjects': len(cox_df),
            'n_events': int(cox_df['event'].sum()),
        }
        with open('data/cox_info.json', 'w') as f:
            json.dump(cox_info, f, indent=2)

        return cph, hazard_ratios

    except ImportError:
        print("lifelines not installed — computing univariate hazard ratios")

        hazard_ratios = []
        for col in covariate_cols:
            median_val = survival_df[col].median()
            high = survival_df[survival_df[col] > median_val]
            low = survival_df[survival_df[col] <= median_val]
            high_rate = high['event'].mean()
            low_rate = max(low['event'].mean(), 0.01)
            hr = high_rate / low_rate
            hazard_ratios.append({
                'feature': col,
                'hazard_ratio': round(hr, 3),
                'high_group_churn_rate': round(high_rate, 3),
                'low_group_churn_rate': round(low_rate, 3),
                'p_value': np.nan,
                'lower_ci': np.nan,
                'upper_ci': np.nan,
            })

        hr_df = pd.DataFrame(hazard_ratios)
        hr_df.to_csv('data/cox_hazard_ratios.csv', index=False)

        cox_info = {
            'concordance_index': None,
            'n_subjects': len(survival_df),
            'n_events': int(survival_df['event'].sum()),
            'note': 'lifelines not available, univariate HRs only',
        }
        with open('data/cox_info.json', 'w') as f:
            json.dump(cox_info, f, indent=2)

        # Still make a plot
        fig, ax = plt.subplots(figsize=(10, 8))
        hr_df_sorted = hr_df.sort_values('hazard_ratio')
        y_pos = range(len(hr_df_sorted))
        colors = ['#e74c3c' if hr > 1 else '#27ae60'
                  for hr in hr_df_sorted['hazard_ratio']]
        ax.barh(y_pos, hr_df_sorted['hazard_ratio'] - 1, left=1,
                color=colors, alpha=0.7, edgecolor='black')
        ax.axvline(1, color='black', linewidth=1)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(
            [f.replace('_', ' ') for f in hr_df_sorted['feature']]
        )
        ax.set_xlabel('Hazard Ratio (above/below median split)')
        ax.set_title('Univariate Hazard Ratios for Driver Churn')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/cox_hazard_ratios.png', dpi=150, bbox_inches='tight')
        plt.close()

        return None, hr_df


# =========================================================
# 4. STACKED ENSEMBLE + MODEL COMPARISON
# =========================================================

def _prepare_task_data(features_df, incidents_df, drivers_df, safety_scores_df):
    """Prepare labeled data for both incident and churn tasks."""
    df = features_df.merge(safety_scores_df, on=['driver_id', 'week'], how='left')
    df['safety_score_trend'] = df.groupby('driver_id')['safety_score'].transform(
        lambda x: x.diff(4)
    )

    # --- Incident labels ---
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

    # --- Churn labels ---
    df['churns_8weeks'] = 0
    for driver_id in df['driver_id'].unique():
        driver = drivers_df[drivers_df['driver_id'] == driver_id]
        if len(driver) == 0:
            continue
        driver = driver.iloc[0]
        if pd.isna(driver['termination_date']):
            continue
        term_week = int(
            pd.to_datetime(driver['termination_date']).isocalendar().week
        )
        mask = df['driver_id'] == driver_id
        df.loc[mask, 'churns_8weeks'] = df.loc[mask, 'week'].apply(
            lambda w: 1 if 0 < (term_week - w) <= 8 else 0
        )

    # --- Burnout score ---
    burnout = np.zeros(len(df))
    burnout += (df['daily_hours'] > 9).astype(int) * 25
    burnout += (df['night_driving_pct'] > 50).astype(int) * 20
    burnout += (df['daily_steps_trend'].fillna(0) < -500).astype(int) * 15
    burnout += (df['total_events_per_100km_trend'].fillna(0) > 0.5).astype(int) * 20
    df['burnout_score'] = burnout.clip(0, 100)

    return df


def train_stacked_ensemble(features_df, incidents_df, drivers_df, safety_scores_df):
    """
    Stacked ensemble: XGBoost + LightGBM + Random Forest + Logistic Regression
    with a Logistic Regression meta-learner.
    Trains for both incident and churn prediction, compares all models.
    """
    df = _prepare_task_data(features_df, incidents_df, drivers_df, safety_scores_df)

    incident_features = [
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

    churn_features = [
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

    results = []

    for task_name, feature_cols, target_col in [
        ('incident', incident_features, 'incident_next_4weeks'),
        ('churn', churn_features, 'churns_8weeks'),
    ]:
        print(f"\n--- Model Comparison: {task_name} ---")

        train_mask = df['week'] <= 20
        test_mask = df['week'] > 20

        X_train = df.loc[train_mask, feature_cols].fillna(0)
        y_train = df.loc[train_mask, target_col]
        X_test = df.loc[test_mask, feature_cols].fillna(0)
        y_test = df.loc[test_mask, target_col]

        n_pos = max((y_train == 1).sum(), 1)
        n_neg = (y_train == 0).sum()
        scale_weight = n_neg / n_pos

        # --- Base models ---
        xgb = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            scale_pos_weight=scale_weight, random_state=42,
            n_jobs=-1, eval_metric='logloss'
        )

        lgbm = LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            scale_pos_weight=scale_weight, random_state=42,
            n_jobs=-1, verbose=-1
        )

        rf = RandomForestClassifier(
            n_estimators=200, max_depth=8,
            class_weight='balanced', random_state=42, n_jobs=-1
        )

        lr = LogisticRegression(
            max_iter=1000, class_weight='balanced',
            random_state=42, C=0.1
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train all base models
        xgb.fit(X_train, y_train)
        lgbm.fit(X_train, y_train)
        rf.fit(X_train, y_train)
        lr.fit(X_train_scaled, y_train)

        xgb_prob = xgb.predict_proba(X_test)[:, 1]
        lgbm_prob = lgbm.predict_proba(X_test)[:, 1]
        rf_prob = rf.predict_proba(X_test)[:, 1]
        lr_prob = lr.predict_proba(X_test_scaled)[:, 1]

        # --- Stacked ensemble ---
        meta_train = np.column_stack([
            xgb.predict_proba(X_train)[:, 1],
            lgbm.predict_proba(X_train)[:, 1],
            rf.predict_proba(X_train)[:, 1],
            lr.predict_proba(X_train_scaled)[:, 1],
        ])
        meta_test = np.column_stack([xgb_prob, lgbm_prob, rf_prob, lr_prob])

        meta_learner = LogisticRegression(max_iter=1000, random_state=42)
        meta_learner.fit(meta_train, y_train)
        ensemble_prob = meta_learner.predict_proba(meta_test)[:, 1]

        if y_test.nunique() > 1:
            xgb_auc = roc_auc_score(y_test, xgb_prob)
            lgbm_auc = roc_auc_score(y_test, lgbm_prob)
            rf_auc = roc_auc_score(y_test, rf_prob)
            lr_auc = roc_auc_score(y_test, lr_prob)
            ensemble_auc = roc_auc_score(y_test, ensemble_prob)

            print(f"  XGBoost AUC:          {xgb_auc:.4f}")
            print(f"  LightGBM AUC:         {lgbm_auc:.4f}")
            print(f"  Random Forest AUC:    {rf_auc:.4f}")
            print(f"  Logistic Reg AUC:     {lr_auc:.4f}")
            print(f"  Stacked Ensemble AUC: {ensemble_auc:.4f}")

            best_individual = max(xgb_auc, lgbm_auc, rf_auc, lr_auc)
            results.append({
                'task': task_name,
                'xgboost_auc': round(xgb_auc, 4),
                'lightgbm_auc': round(lgbm_auc, 4),
                'random_forest_auc': round(rf_auc, 4),
                'logistic_regression_auc': round(lr_auc, 4),
                'stacked_ensemble_auc': round(ensemble_auc, 4),
                'ensemble_lift': round(
                    (ensemble_auc - best_individual) / max(best_individual, 1e-9) * 100, 2
                ),
            })

            # ROC curves comparison plot
            fig, ax = plt.subplots(figsize=(10, 8))
            for name, probs, color, ls in [
                ('XGBoost', xgb_prob, '#e74c3c', '-'),
                ('LightGBM', lgbm_prob, '#f39c12', '-'),
                ('Random Forest', rf_prob, '#3498db', '-'),
                ('Logistic Regression', lr_prob, '#2ecc71', '--'),
                ('Stacked Ensemble', ensemble_prob, '#9b59b6', '-'),
            ]:
                fpr, tpr, _ = roc_curve(y_test, probs)
                auc = roc_auc_score(y_test, probs)
                ax.plot(fpr, tpr, color=color, linewidth=2,
                        linestyle=ls, label=f'{name} (AUC={auc:.3f})')

            ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'Model Comparison: {task_name.title()} Prediction')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'results/ensemble_roc_{task_name}.png',
                        dpi=150, bbox_inches='tight')
            plt.close()

        # Save ensemble artifacts
        joblib.dump({
            'xgboost': xgb,
            'lightgbm': lgbm,
            'random_forest': rf,
            'logistic_regression': lr,
            'scaler': scaler,
            'meta_learner': meta_learner,
            'feature_cols': feature_cols,
        }, f'models/ensemble_{task_name}.joblib')

    results_df = pd.DataFrame(results)
    results_df.to_csv('data/ensemble_comparison.csv', index=False)
    print(f"\nEnsemble comparison saved")

    return results_df


# =========================================================
# 5. HYPERPARAMETER TUNING — Optuna + Time-Series CV
# =========================================================

def tune_hyperparameters(features_df, incidents_df, drivers_df, safety_scores_df):
    """
    Bayesian hyperparameter optimization using Optuna with
    expanding-window time-series cross-validation.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    df = _prepare_task_data(features_df, incidents_df, drivers_df, safety_scores_df)

    feature_cols = [
        'hard_brakes_per_100km', 'hard_accels_per_100km',
        'speeding_per_100km', 'total_events_per_100km',
        'idle_time_pct', 'daily_hours', 'daily_km',
        'trips_per_day', 'night_driving_pct',
        'daily_steps', 'daily_floors', 'daily_lifting_events',
        'avg_package_weight_kg',
        'age', 'experience_years',
        'has_eco_training', 'has_safety_training',
        'safety_score', 'safety_score_trend',
        'burnout_score',
        'total_events_per_100km_trend', 'fuel_per_100km_trend',
        'daily_hours_trend', 'daily_steps_trend', 'co2_per_km_trend',
    ]

    # Expanding-window time-series CV
    cv_folds = [
        (df['week'] <= 8,  (df['week'] > 8)  & (df['week'] <= 12)),
        (df['week'] <= 12, (df['week'] > 12) & (df['week'] <= 16)),
        (df['week'] <= 16, (df['week'] > 16) & (df['week'] <= 20)),
    ]

    n_pos = max((df[df['week'] <= 20]['churns_8weeks'] == 1).sum(), 1)
    n_neg = (df[df['week'] <= 20]['churns_8weeks'] == 0).sum()
    base_scale_weight = n_neg / n_pos

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'scale_pos_weight': base_scale_weight,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'logloss',
        }

        aucs = []
        for train_mask, val_mask in cv_folds:
            X_t = df.loc[train_mask, feature_cols].fillna(0)
            y_t = df.loc[train_mask, 'churns_8weeks']
            X_v = df.loc[val_mask, feature_cols].fillna(0)
            y_v = df.loc[val_mask, 'churns_8weeks']

            model = XGBClassifier(**params)
            model.fit(X_t, y_t)

            y_prob = model.predict_proba(X_v)[:, 1]
            if y_v.nunique() > 1:
                aucs.append(roc_auc_score(y_v, y_prob))

        return np.mean(aucs) if aucs else 0

    study = optuna.create_study(direction='maximize', study_name='churn_tuning')
    study.optimize(objective, n_trials=50, show_progress_bar=False)

    print(f"\nBest trial:")
    print(f"  AUC (CV): {study.best_trial.value:.4f}")
    print(f"  Params: {study.best_trial.params}")

    # Final evaluation: default vs tuned on held-out test set
    best_params = study.best_trial.params.copy()
    best_params['scale_pos_weight'] = base_scale_weight
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1
    best_params['eval_metric'] = 'logloss'

    train_mask = df['week'] <= 20
    test_mask = df['week'] > 20

    X_train = df.loc[train_mask, feature_cols].fillna(0)
    y_train = df.loc[train_mask, 'churns_8weeks']
    X_test = df.loc[test_mask, feature_cols].fillna(0)
    y_test = df.loc[test_mask, 'churns_8weeks']

    # Default model
    default_model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        scale_pos_weight=base_scale_weight, random_state=42,
        n_jobs=-1, eval_metric='logloss'
    )
    default_model.fit(X_train, y_train)
    default_prob = default_model.predict_proba(X_test)[:, 1]
    default_auc = roc_auc_score(y_test, default_prob) if y_test.nunique() > 1 else 0

    # Tuned model
    tuned_model = XGBClassifier(**best_params)
    tuned_model.fit(X_train, y_train)
    tuned_prob = tuned_model.predict_proba(X_test)[:, 1]
    tuned_auc = roc_auc_score(y_test, tuned_prob) if y_test.nunique() > 1 else 0

    print(f"\nDefault AUC: {default_auc:.4f}")
    print(f"Tuned AUC:   {tuned_auc:.4f}")
    improvement = (tuned_auc - default_auc) / max(default_auc, 1e-9) * 100
    print(f"Improvement: {improvement:.1f}%")

    joblib.dump(tuned_model, 'models/tuned_churn_model.joblib')

    # Save tuning history
    trials_data = []
    for trial in study.trials:
        trials_data.append({
            'trial': trial.number,
            'auc_cv': round(trial.value, 4),
            **trial.params,
        })
    trials_df = pd.DataFrame(trials_data)
    trials_df.to_csv('data/tuning_history.csv', index=False)

    # Save comparison
    tuning_comparison = {
        'default_auc': round(default_auc, 4),
        'tuned_auc': round(tuned_auc, 4),
        'improvement_pct': round(improvement, 2),
        'best_cv_auc': round(study.best_trial.value, 4),
        'n_trials': len(study.trials),
        'best_params': {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in study.best_trial.params.items()
        },
    }
    with open('data/tuning_results.json', 'w') as f:
        json.dump(tuning_comparison, f, indent=2)

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1: Optimization convergence
    ax = axes[0]
    trial_nums = [t.number for t in study.trials]
    trial_values = [t.value for t in study.trials]
    ax.scatter(trial_nums, trial_values, alpha=0.6, c='steelblue', s=30)
    running_best = []
    best_so_far = 0
    for v in trial_values:
        best_so_far = max(best_so_far, v)
        running_best.append(best_so_far)
    ax.plot(trial_nums, running_best, 'r-', linewidth=2, label='Best so far')
    ax.set_xlabel('Trial')
    ax.set_ylabel('AUC (CV)')
    ax.set_title('Optuna Optimization History')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2: Parameter importance (correlation-based)
    ax = axes[1]
    importances = {}
    for param in ['n_estimators', 'max_depth', 'learning_rate',
                   'subsample', 'colsample_bytree',
                   'min_child_weight', 'reg_alpha', 'reg_lambda']:
        param_vals = [t.params.get(param, 0) for t in study.trials]
        trial_vals = [t.value for t in study.trials]
        if len(set(param_vals)) > 1:
            corr = abs(np.corrcoef(param_vals, trial_vals)[0, 1])
            if not np.isnan(corr):
                importances[param] = corr

    if importances:
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        ax.barh(
            [x[0].replace('_', ' ') for x in sorted_imp],
            [x[1] for x in sorted_imp],
            color='steelblue', edgecolor='black', alpha=0.7
        )
        ax.set_xlabel('Correlation with AUC')
        ax.set_title('Hyperparameter Importance')
        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/tuning_history.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Default vs Tuned ROC comparison
    if y_test.nunique() > 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        for name, probs, color in [
            ('Default XGBoost', default_prob, '#3498db'),
            ('Tuned XGBoost (Optuna)', tuned_prob, '#e74c3c'),
        ]:
            fpr, tpr, _ = roc_curve(y_test, probs)
            auc = roc_auc_score(y_test, probs)
            ax.plot(fpr, tpr, color=color, linewidth=2,
                    label=f'{name} (AUC={auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Default vs Tuned XGBoost (Churn Prediction)')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/tuning_roc_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

    return study, tuned_model


if __name__ == '__main__':
    print("Loading data...")
    features_df = pd.read_csv('data/weekly_features.csv')
    drivers_df = pd.read_csv('data/drivers.csv')
    safety_scores_df = pd.read_csv('data/driver_safety_scores.csv')
    incidents_df = pd.read_csv('data/incidents.csv')

    print("\n=== 1. ANOMALY DETECTION ===")
    train_anomaly_detector(features_df)

    print("\n=== 2. DRIVER CLUSTERING ===")
    cluster_drivers(features_df)

    print("\n=== 3. COX PROPORTIONAL HAZARDS ===")
    train_cox_model(drivers_df, features_df, safety_scores_df)

    print("\n=== 4. STACKED ENSEMBLE + MODEL COMPARISON ===")
    train_stacked_ensemble(features_df, incidents_df, drivers_df, safety_scores_df)

    print("\n=== 5. HYPERPARAMETER TUNING ===")
    tune_hyperparameters(features_df, incidents_df, drivers_df, safety_scores_df)

    print("\n=== ALL ADVANCED MODELS COMPLETE ===")
