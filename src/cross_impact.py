"""
Cross-Impact Analysis
The unique differentiator — shows which interventions improve
multiple outcomes simultaneously.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('results', exist_ok=True)


def compute_cross_correlations(features_df):
    """Compute correlations between key metrics across all driver-weeks."""
    # Merge safety scores if available
    try:
        safety_df = pd.read_csv('data/driver_safety_scores.csv')
        df = features_df.merge(safety_df, on=['driver_id', 'week'], how='left')
    except FileNotFoundError:
        df = features_df.copy()
        df['safety_score'] = 50  # placeholder

    key_metrics = [
        'co2_per_km', 'fuel_per_100km', 'idle_fuel_pct',
        'total_events_per_100km', 'hard_brakes_per_100km',
        'daily_hours', 'night_driving_pct', 'daily_steps',
        'daily_floors',
    ]

    # Add safety_score if available
    if 'safety_score' in df.columns:
        key_metrics.insert(5, 'safety_score')

    available = [c for c in key_metrics if c in df.columns]
    correlation_matrix = df[available].corr()

    # Save correlation heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(correlation_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.8)

    labels = [c.replace('_', '\n') for c in available]
    ax.set_xticks(range(len(available)))
    ax.set_yticks(range(len(available)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(len(available)):
        for j in range(len(available)):
            val = correlation_matrix.values[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                   fontsize=8, color=color)

    ax.set_title('Cross-Impact Correlation Matrix\n(Carbon × Safety × Workload)',
                fontsize=14)
    plt.tight_layout()
    plt.savefig('results/cross_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Cross-correlation heatmap saved")
    return correlation_matrix


def build_intervention_matrix():
    """
    Build the intervention simulation matrix.
    Pre-computed impacts for each intervention on each outcome.
    """
    interventions = [
        {
            'intervention': 'Eco-driving coaching',
            'description': 'Train worst 30% drivers on eco-driving techniques',
            'annual_cost': 30000,
            'carbon_impact_pct': -18,
            'safety_impact_pct': -12,
            'churn_impact_pct': -8,
            'injury_impact_pct': -5,
            'carbon_savings_usd': 45000,
            'incident_savings_usd': 38000,
            'retention_savings_usd': 24000,
        },
        {
            'intervention': 'Shift schedule fix',
            'description': 'Convert rotating shifts to fixed day/night',
            'annual_cost': 10000,
            'carbon_impact_pct': -3,
            'safety_impact_pct': -25,
            'churn_impact_pct': -30,
            'injury_impact_pct': -15,
            'carbon_savings_usd': 8000,
            'incident_savings_usd': 52000,
            'retention_savings_usd': 66000,
        },
        {
            'intervention': 'Route rebalancing',
            'description': 'Equalize physical workload across routes',
            'annual_cost': 5000,
            'carbon_impact_pct': -2,
            'safety_impact_pct': -5,
            'churn_impact_pct': -25,
            'injury_impact_pct': -20,
            'carbon_savings_usd': 3000,
            'incident_savings_usd': 12000,
            'retention_savings_usd': 54000,
        },
        {
            'intervention': 'Anti-idle auto-off',
            'description': 'Install automatic engine shutoff after 60s idle',
            'annual_cost': 50000,
            'carbon_impact_pct': -15,
            'safety_impact_pct': 0,
            'churn_impact_pct': 0,
            'injury_impact_pct': 0,
            'carbon_savings_usd': 62000,
            'incident_savings_usd': 0,
            'retention_savings_usd': 0,
        },
        {
            'intervention': '30% EV replacement',
            'description': 'Replace 30% of diesel fleet with electric vans',
            'annual_cost': 900000,
            'carbon_impact_pct': -30,
            'safety_impact_pct': -2,
            'churn_impact_pct': -3,
            'injury_impact_pct': 0,
            'carbon_savings_usd': 180000,
            'incident_savings_usd': 5000,
            'retention_savings_usd': 15000,
        },
    ]

    for inv in interventions:
        total_savings = (
            inv['carbon_savings_usd'] +
            inv['incident_savings_usd'] +
            inv['retention_savings_usd']
        )
        inv['total_savings_usd'] = total_savings
        inv['net_roi'] = round(total_savings / max(inv['annual_cost'], 1), 1)

    df = pd.DataFrame(interventions)
    df.to_csv('data/intervention_roi.csv', index=False)

    # Cross-impact matrix (intervention x outcome)
    impact_data = []
    for inv in interventions:
        impact_data.append({
            'intervention': inv['intervention'],
            'carbon': inv['carbon_impact_pct'],
            'safety': inv['safety_impact_pct'],
            'retention': inv['churn_impact_pct'],
            'injury': inv['injury_impact_pct'],
        })

    impact_df = pd.DataFrame(impact_data)
    impact_df.to_csv('data/cross_impact_matrix.csv', index=False)

    # Intervention comparison chart
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(impact_df))
    width = 0.2

    categories = ['carbon', 'safety', 'retention', 'injury']
    colors = ['#27ae60', '#2980b9', '#e67e22', '#c0392b']
    labels = ['Carbon', 'Safety', 'Retention', 'Injury']

    for i, (cat, color, label) in enumerate(zip(categories, colors, labels)):
        ax.bar(x + i * width, impact_df[cat], width,
               label=label, color=color, edgecolor='black', alpha=0.85)

    ax.set_xlabel('Intervention', fontsize=12)
    ax.set_ylabel('Impact (%)', fontsize=12)
    ax.set_title('Cross-Impact: How Each Intervention Affects All Outcomes', fontsize=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(impact_df['intervention'], rotation=15, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/intervention_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Intervention comparison saved")
    print(f"\nIntervention ROI Rankings:")
    for _, row in df.sort_values('net_roi', ascending=False).iterrows():
        print(f"  {row['intervention']:25s} ROI: {row['net_roi']:5.1f}x "
              f"(cost: ${row['annual_cost']:>9,}, saves: ${row['total_savings_usd']:>9,})")

    return df


if __name__ == '__main__':
    print("Loading features...")
    features_df = pd.read_csv('data/weekly_features.csv')

    print("\n--- Computing Cross-Correlations ---")
    compute_cross_correlations(features_df)

    print("\n--- Building Intervention Matrix ---")
    build_intervention_matrix()

    print("\nCross-impact analysis complete.")
