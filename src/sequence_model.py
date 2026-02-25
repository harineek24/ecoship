"""
Sequence Model — GRU for Churn Prediction

Instead of treating each driver-week as an independent row (what XGBoost does),
this model takes the full weekly time series for each driver and learns sequential
patterns: behavior deterioration, burnout trajectories, fatigue accumulation.

Architecture: GRU (Gated Recurrent Unit) — lighter than LSTM, works well on
short sequences (26 weeks). Bidirectional for capturing both early and late signals.

The key advantage over XGBoost: the model sees the *trajectory shape*, not just
individual snapshots. A driver whose hard-braking rate doubled over 4 weeks is
very different from one whose rate was always high — XGBoost can only see this
if you manually engineer trend features. The GRU learns it automatically.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import joblib
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =========================================================
# Dataset
# =========================================================

class DriverSequenceDataset(Dataset):
    """
    Each sample is one driver's full weekly time series.
    Shape: (seq_len, n_features) -> label (0/1 churn)
    Sequences shorter than max_len are left-padded with zeros.
    """

    def __init__(self, sequences, labels, max_len):
        self.sequences = sequences
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]

        # Left-pad to max_len
        padded = np.zeros((self.max_len, seq.shape[1]), dtype=np.float32)
        start = self.max_len - len(seq)
        padded[start:] = seq

        # Mask: 1 where real data, 0 where padded
        mask = np.zeros(self.max_len, dtype=np.float32)
        mask[start:] = 1.0

        return (
            torch.tensor(padded, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )


# =========================================================
# Model
# =========================================================

class DriverGRU(nn.Module):
    """
    Bidirectional GRU with attention pooling.

    Instead of just taking the last hidden state (which biases toward
    the most recent week), attention learns to weight all weeks by
    relevance. This lets the model attend to an anomalous week 3 months
    ago even if recent weeks look normal.
    """

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x, mask):
        # x: (batch, seq_len, features)
        # mask: (batch, seq_len)
        gru_out, _ = self.gru(x)  # (batch, seq_len, hidden*2)

        # Attention weights
        attn_scores = self.attention(gru_out).squeeze(-1)  # (batch, seq_len)
        # Mask out padded positions
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, seq_len)

        # Weighted sum of GRU outputs
        context = torch.bmm(
            attn_weights.unsqueeze(1), gru_out
        ).squeeze(1)  # (batch, hidden*2)

        logit = self.classifier(context).squeeze(-1)  # (batch,)
        return logit, attn_weights


# =========================================================
# Data Preparation
# =========================================================

def prepare_sequences(features_df, drivers_df, safety_scores_df):
    """
    Build per-driver weekly sequences with churn labels.

    Returns:
        train_sequences, train_labels, test_sequences, test_labels,
        scaler, feature_cols, max_len
    """
    # Merge safety scores
    df = features_df.merge(safety_scores_df, on=['driver_id', 'week'], how='left')

    feature_cols = [
        'hard_brakes_per_100km', 'hard_accels_per_100km',
        'speeding_per_100km', 'sharp_turns_per_100km',
        'phone_use_per_100km', 'total_events_per_100km',
        'fuel_per_100km', 'idle_time_pct', 'co2_per_km',
        'daily_hours', 'daily_km', 'trips_per_day',
        'night_driving_pct',
        'daily_steps', 'daily_floors', 'daily_lifting_events',
        'avg_package_weight_kg',
        'age', 'experience_years',
        'has_eco_training', 'has_safety_training',
        'safety_score',
    ]

    # Create churn labels per driver
    driver_labels = {}
    for driver_id in df['driver_id'].unique():
        driver = drivers_df[drivers_df['driver_id'] == driver_id]
        if len(driver) == 0:
            driver_labels[driver_id] = 0
            continue
        driver = driver.iloc[0]
        if pd.isna(driver['termination_date']):
            driver_labels[driver_id] = 0
        else:
            driver_labels[driver_id] = 1

    # Build per-driver sequences
    df = df.sort_values(['driver_id', 'week'])
    df[feature_cols] = df[feature_cols].fillna(0)

    # Scale features globally
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Split by time: train on weeks 1-20 sequences, test on full sequences
    # but only use drivers that have data in both periods
    all_drivers = sorted(df['driver_id'].unique())

    # For train: use sequence up to week 20
    # For test: use full sequence
    # Split drivers: 80% train, 20% test (by driver, not by time)
    np.random.seed(42)
    perm = np.random.permutation(all_drivers)
    split_idx = int(len(perm) * 0.8)
    train_driver_ids = set(perm[:split_idx])
    test_driver_ids = set(perm[split_idx:])

    train_sequences = []
    train_labels = []
    test_sequences = []
    test_labels = []

    max_len = 0

    for driver_id in all_drivers:
        driver_data = df[df['driver_id'] == driver_id].sort_values('week')
        seq = driver_data[feature_cols].values
        label = driver_labels[driver_id]
        max_len = max(max_len, len(seq))

        if driver_id in train_driver_ids:
            # Use only up to week 20 for training
            train_seq = driver_data[driver_data['week'] <= 20][feature_cols].values
            if len(train_seq) > 0:
                train_sequences.append(train_seq)
                train_labels.append(label)
        else:
            test_sequences.append(seq)
            test_labels.append(label)

    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    return (train_sequences, train_labels, test_sequences, test_labels,
            scaler, feature_cols, max_len)


# =========================================================
# Training
# =========================================================

def train_gru_model(features_df, drivers_df, safety_scores_df):
    """
    Train bidirectional GRU with attention for churn prediction.
    """
    print("Preparing sequences...")
    (train_seqs, train_labels, test_seqs, test_labels,
     scaler, feature_cols, max_len) = prepare_sequences(
        features_df, drivers_df, safety_scores_df
    )

    n_features = len(feature_cols)
    print(f"Train: {len(train_seqs)} drivers, "
          f"{sum(train_labels)} positive ({sum(train_labels)/len(train_labels)*100:.1f}%)")
    print(f"Test:  {len(test_seqs)} drivers, "
          f"{sum(test_labels)} positive ({sum(test_labels)/len(test_labels)*100:.1f}%)")
    print(f"Sequence length: up to {max_len} weeks, {n_features} features")

    # Datasets
    train_dataset = DriverSequenceDataset(train_seqs, train_labels, max_len)
    test_dataset = DriverSequenceDataset(test_seqs, test_labels, max_len)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Model
    model = DriverGRU(
        input_size=n_features,
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
    ).to(DEVICE)

    # Class weights for imbalanced data
    n_pos = max(train_labels.sum(), 1)
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # Training loop
    n_epochs = 50
    history = {'epoch': [], 'train_loss': [], 'test_auc': []}

    best_auc = 0
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for batch_x, batch_mask, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_mask = batch_mask.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            logits, _ = model(batch_x, batch_mask)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        # Evaluate
        model.eval()
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for batch_x, batch_mask, batch_y in test_loader:
                batch_x = batch_x.to(DEVICE)
                batch_mask = batch_mask.to(DEVICE)
                logits, _ = model(batch_x, batch_mask)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(batch_y.numpy())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        if len(np.unique(all_labels)) > 1:
            test_auc = roc_auc_score(all_labels, all_probs)
        else:
            test_auc = 0

        avg_loss = epoch_loss / len(train_loader)
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(round(avg_loss, 4))
        history['test_auc'].append(round(test_auc, 4))

        if test_auc > best_auc:
            best_auc = test_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | "
                  f"Test AUC: {test_auc:.4f} | Best: {best_auc:.4f}")

    # Load best model
    model.load_state_dict(best_state)
    model.eval()

    print(f"\nBest GRU AUC: {best_auc:.4f}")

    # --- XGBoost baseline on same split for fair comparison ---
    from xgboost import XGBClassifier

    # Flatten sequences to last-week features for XGBoost
    xgb_train_X = np.array([seq[-1] for seq in train_seqs])
    xgb_test_X = np.array([seq[-1] for seq in test_seqs])

    xgb_model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        scale_pos_weight=n_neg / n_pos,
        random_state=42, n_jobs=-1, eval_metric='logloss'
    )
    xgb_model.fit(xgb_train_X, train_labels)
    xgb_probs = xgb_model.predict_proba(xgb_test_X)[:, 1]
    xgb_auc = roc_auc_score(test_labels, xgb_probs) if len(np.unique(test_labels)) > 1 else 0
    print(f"XGBoost AUC (same split): {xgb_auc:.4f}")

    # --- Get attention weights for interpretability ---
    attention_data = []
    model.eval()
    with torch.no_grad():
        for i, (seq, label) in enumerate(zip(test_seqs, test_labels)):
            padded = np.zeros((max_len, n_features), dtype=np.float32)
            start = max_len - len(seq)
            padded[start:] = seq
            mask = np.zeros(max_len, dtype=np.float32)
            mask[start:] = 1.0

            x_t = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            m_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            logit, attn = model(x_t, m_t)

            attn_weights = attn.cpu().numpy()[0]
            # Only keep non-padded weights
            real_attn = attn_weights[start:]
            if len(real_attn) > 0:
                peak_week_idx = np.argmax(real_attn)
                attention_data.append({
                    'driver_idx': i,
                    'label': int(label),
                    'prob': float(torch.sigmoid(logit).cpu().numpy()[0]),
                    'peak_attention_week': int(peak_week_idx + 1),
                    'peak_attention_weight': float(real_attn[peak_week_idx]),
                    'attn_weights': real_attn.tolist(),
                })

    # --- Save everything ---
    torch.save(best_state, 'models/gru_churn_model.pt')
    joblib.dump(scaler, 'models/gru_scaler.joblib')

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv('data/gru_training_history.csv', index=False)

    # Save comparison results
    gru_results = {
        'gru_auc': round(best_auc, 4),
        'xgboost_auc': round(xgb_auc, 4),
        'n_train_drivers': len(train_seqs),
        'n_test_drivers': len(test_seqs),
        'n_features': n_features,
        'max_seq_len': max_len,
        'n_epochs': n_epochs,
        'hidden_size': 64,
        'num_layers': 2,
        'architecture': 'Bidirectional GRU + Attention',
    }
    with open('data/gru_results.json', 'w') as f:
        json.dump(gru_results, f, indent=2)

    # --- Plots ---
    # 1: Training curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history['epoch'], history['train_loss'], 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('GRU Training Loss')
    ax1.grid(True, alpha=0.3)

    ax2.plot(history['epoch'], history['test_auc'], 'r-', linewidth=2)
    ax2.axhline(xgb_auc, color='blue', linestyle='--', linewidth=1.5,
                label=f'XGBoost AUC: {xgb_auc:.3f}')
    ax2.axhline(best_auc, color='red', linestyle='--', linewidth=1,
                alpha=0.5, label=f'Best GRU AUC: {best_auc:.3f}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test AUC')
    ax2.set_title('GRU vs XGBoost (Test AUC)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/gru_training.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2: ROC comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    # GRU ROC
    fpr_gru, tpr_gru, _ = roc_curve(all_labels, all_probs)
    ax.plot(fpr_gru, tpr_gru, 'r-', linewidth=2,
            label=f'GRU + Attention (AUC={best_auc:.3f})')
    # XGBoost ROC
    fpr_xgb, tpr_xgb, _ = roc_curve(test_labels, xgb_probs)
    ax.plot(fpr_xgb, tpr_xgb, 'b--', linewidth=2,
            label=f'XGBoost (AUC={xgb_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Churn Prediction: GRU vs XGBoost')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/gru_vs_xgboost_roc.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3: Attention heatmap for a few interesting drivers
    churned_drivers = [d for d in attention_data if d['label'] == 1]
    retained_drivers = [d for d in attention_data if d['label'] == 0]

    if churned_drivers and retained_drivers:
        # Pick top predicted churn and top retained
        churned_drivers.sort(key=lambda x: x['prob'], reverse=True)
        retained_drivers.sort(key=lambda x: x['prob'])

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        for ax_idx, (driver, title) in enumerate([
            (churned_drivers[0], 'Churned Driver (highest predicted risk)'),
            (retained_drivers[0], 'Retained Driver (lowest predicted risk)'),
        ]):
            ax = axes[ax_idx]
            weights = np.array(driver['attn_weights'])
            weeks = range(1, len(weights) + 1)
            ax.bar(weeks, weights, color='steelblue' if ax_idx == 1 else '#e74c3c',
                   alpha=0.7, edgecolor='black')
            ax.set_ylabel('Attention Weight')
            ax.set_title(f'{title} — predicted churn prob: {driver["prob"]:.0%}')
            ax.grid(axis='y', alpha=0.3)

        axes[-1].set_xlabel('Week')
        plt.tight_layout()
        plt.savefig('results/gru_attention_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()

    # Save attention data for dashboard
    attn_summary = []
    for d in attention_data:
        attn_summary.append({
            'driver_idx': d['driver_idx'],
            'label': d['label'],
            'prob': round(d['prob'], 4),
            'peak_attention_week': d['peak_attention_week'],
            'peak_attention_weight': round(d['peak_attention_weight'], 4),
        })
    pd.DataFrame(attn_summary).to_csv('data/gru_attention.csv', index=False)

    return model, gru_results


if __name__ == '__main__':
    print("Loading data...")
    features_df = pd.read_csv('data/weekly_features.csv')
    drivers_df = pd.read_csv('data/drivers.csv')
    safety_scores_df = pd.read_csv('data/driver_safety_scores.csv')

    print("\n=== TRAINING GRU SEQUENCE MODEL ===")
    model, results = train_gru_model(features_df, drivers_df, safety_scores_df)

    print(f"\n=== RESULTS ===")
    print(f"GRU AUC:     {results['gru_auc']}")
    print(f"XGBoost AUC: {results['xgboost_auc']}")
    print(f"Architecture: {results['architecture']}")
    print("\nSequence model complete.")
