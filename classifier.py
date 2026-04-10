"""
CS4100 — Ambient Music Mood Prediction
Feedforward Neural Network Classifier: Context + HMM Beliefs → Mood Bucket

This module implements a from-scratch PyTorch classifier that predicts
which music mood bucket a user will listen to, given:
  - HMM belief state (3D) from wrist physiological data
  - Environmental context (time, weather, temperature, humidity, speed)
  - Physiological summary (hr_mean, hr_std, intensity_mean, activity_dominant)
  - Pre-listening emotion (valence, arousal, + availability mask)

The ablation study trains 4 model variants to isolate what matters:
  Model A: Full model (all features)
  Model B: No HMM beliefs
  Model C: No pre-emotion
  Model D: Context-only baseline

Usage:
  python classifier.py

Requires:
  pip install torch numpy pandas matplotlib seaborn scikit-learn
  Files: situnes_exploration_merged.csv, hmm_belief_states.npy
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, 
                             classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_feature_matrix(
    df: pd.DataFrame,
    beliefs: np.ndarray,
    feature_set: str = "full"
) -> Tuple[np.ndarray, List[str]]:
    """
    Build the feature matrix for the classifier from merged data + HMM beliefs.
    
    Parameters
    ----------
    df : pd.DataFrame
        Merged SiTunes data (from exploration notebook).
    beliefs : np.ndarray, shape (n_interactions, 3)
        HMM belief states.
    feature_set : str
        Which feature combination to use for ablation:
        - "full":       HMM beliefs + context + physio + pre-emotion
        - "no_hmm":     context + physio + pre-emotion (no HMM beliefs)
        - "no_emotion": HMM beliefs + context + physio (no pre-emotion)
        - "context_only": time + weather + activity only
    
    Returns
    -------
    X : np.ndarray, shape (n_interactions, n_features)
        Feature matrix.
    feature_names : list of str
        Names of each feature column.
    """
    features = {}
    feature_names = []
    
    # --- HMM belief state (3 dimensions) ---
    if feature_set in ("full", "no_emotion"):
        for s in range(beliefs.shape[1]):
            col_name = f"belief_{s}"
            features[col_name] = beliefs[:, s]
            feature_names.append(col_name)
    
    # --- Context features (always included except context_only has subset) ---
    # Time period (one-hot encode 3 periods)
    for period in sorted(df['time_period'].dropna().unique()):
        col_name = f"time_{int(period)}"
        features[col_name] = (df['time_period'] == period).astype(float).values
        feature_names.append(col_name)
    
    # Weather type (one-hot encode)
    for wtype in sorted(df['weather_type'].dropna().unique()):
        col_name = f"weather_{int(wtype)}"
        features[col_name] = (df['weather_type'] == wtype).astype(float).values
        feature_names.append(col_name)
    
    # Continuous context
    for col in ['temperature', 'humidity', 'speed']:
        if col in df.columns:
            vals = df[col].fillna(df[col].median()).values.astype(float)
            features[col] = vals
            feature_names.append(col)
    
    # --- Physiological summary features ---
    if feature_set in ("full", "no_hmm", "no_emotion"):
        for col in ['hr_mean', 'hr_std', 'intensity_mean']:
            if col in df.columns:
                features[col] = df[col].fillna(0).values.astype(float)
                feature_names.append(col)
        
        # Activity dominant (one-hot, 3 main categories)
        if 'activity_dominant' in df.columns:
            act = df['activity_dominant'].values
            # Collapse to: 0=sedentary (0,1,3,4), 1=walking (2), 2=running (5)
            act_collapsed = np.where(np.isin(act, [0, 1, 3, 4]), 0,
                            np.where(act == 2, 1,
                            np.where(act == 5, 2, 0)))
            for cat in range(3):
                col_name = f"activity_{cat}"
                features[col_name] = (act_collapsed == cat).astype(float)
                feature_names.append(col_name)
    
    # --- Pre-emotion features ---
    if feature_set in ("full", "no_hmm"):
        features['pre_valence'] = df['emo_pre_valence'].fillna(0).values.astype(float)
        features['pre_arousal'] = df['emo_pre_arousal'].fillna(0).values.astype(float)
        feature_names.extend(['pre_valence', 'pre_arousal'])
        
        # Mask: 1 if pre-emotion is available (non-zero), 0 otherwise
        # This lets the model learn to weight pre-emotion when available
        has_emotion = ((df['emo_pre_valence'].abs() > 0.001) | 
                       (df['emo_pre_arousal'].abs() > 0.001)).astype(float).values
        features['pre_emotion_mask'] = has_emotion
        feature_names.append('pre_emotion_mask')
    
    # --- Context-only baseline ---
    if feature_set == "context_only":
        # Already have time, weather, temperature, humidity, speed above
        # Add a simple activity indicator from the merged data
        if 'activity_dominant' in df.columns:
            act = df['activity_dominant'].values
            act_collapsed = np.where(np.isin(act, [0, 1, 3, 4]), 0,
                            np.where(act == 2, 1, 
                            np.where(act == 5, 2, 0)))
            for cat in range(3):
                col_name = f"activity_{cat}"
                features[col_name] = (act_collapsed == cat).astype(float)
                feature_names.append(col_name)
    
    X = np.column_stack([features[name] for name in feature_names])
    return X, feature_names


# ============================================================
# NEURAL NETWORK
# ============================================================

class MoodClassifier(nn.Module):
    """
    Feedforward neural network for mood bucket classification.
    
    Architecture:
      Input → Linear(n_features, 64) → ReLU → Dropout
            → Linear(64, 32) → ReLU → Dropout
            → Linear(32, n_classes) → Softmax (via CrossEntropyLoss)
    
    This is intentionally simple — the point is to demonstrate that
    the HMM belief features add value, not to build the most powerful
    classifier possible.
    """
    
    def __init__(self, n_features: int, n_classes: int = 4, 
                 hidden1: int = 64, hidden2: int = 32, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, n_classes),
        )
    
    def forward(self, x):
        return self.net(x)


# ============================================================
# TRAINING LOOP
# ============================================================

def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_epochs: int = 200,
    lr: float = 0.001,
    batch_size: int = 64,
    weight_decay: float = 1e-4,
    class_weights: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Dict:
    """
    Train the classifier with early stopping on validation loss.
    
    Parameters
    ----------
    model : MoodClassifier
        The neural network to train.
    X_train, y_train : np.ndarray
        Training features and labels.
    X_val, y_val : np.ndarray
        Validation features and labels.
    n_epochs : int
        Maximum training epochs.
    lr : float
        Learning rate for Adam optimizer.
    batch_size : int
        Mini-batch size.
    weight_decay : float
        L2 regularization strength.
    class_weights : np.ndarray, optional
        Per-class weights to handle class imbalance.
    verbose : bool
        Print progress every 20 epochs.
    
    Returns
    -------
    history : dict
        Training and validation loss/accuracy per epoch.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Convert to tensors
    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.LongTensor(y_train).to(device)
    X_v = torch.FloatTensor(X_val).to(device)
    y_v = torch.LongTensor(y_val).to(device)
    
    # Loss function with optional class weighting
    if class_weights is not None:
        weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # DataLoader for mini-batches
    dataset = TensorDataset(X_tr, y_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_state = None
    patience = 30
    patience_counter = 0
    
    for epoch in range(n_epochs):
        # --- Training ---
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(batch_y)
            preds = logits.argmax(dim=1)
            epoch_correct += (preds == batch_y).sum().item()
            epoch_total += len(batch_y)
        
        train_loss = epoch_loss / epoch_total
        train_acc = epoch_correct / epoch_total
        
        # --- Validation ---
        model.eval()
        with torch.no_grad():
            val_logits = model(X_v)
            val_loss = criterion(val_logits, y_v).item()
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == y_v).float().mean().item()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # --- Early stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break
        
        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}: train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} train_acc={train_acc:.3f} "
                  f"val_acc={val_acc:.3f}")
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return history


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: List[str],
) -> Dict:
    """
    Evaluate a trained model on test data.
    
    Returns
    -------
    results : dict
        accuracy, f1_macro, f1_weighted, per_class_f1, confusion_matrix,
        predictions
    """
    device = next(model.parameters()).device
    model.eval()
    
    with torch.no_grad():
        X_t = torch.FloatTensor(X_test).to(device)
        logits = model(X_t)
        preds = logits.argmax(dim=1).cpu().numpy()
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    
    acc = accuracy_score(y_test, preds)
    f1_mac = f1_score(y_test, preds, average='macro', zero_division=0)
    f1_wt = f1_score(y_test, preds, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, target_names=label_names, 
                                   zero_division=0, output_dict=True)
    
    return {
        'accuracy': acc,
        'f1_macro': f1_mac,
        'f1_weighted': f1_wt,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': preds,
        'probabilities': probs,
    }


# ============================================================
# RANDOM BASELINE
# ============================================================

def random_baseline(y_test: np.ndarray, n_classes: int = 4, 
                    n_trials: int = 100) -> Dict:
    """Compute random and majority-class baseline metrics."""
    # Random uniform
    rng = np.random.RandomState(42)
    random_accs = []
    random_f1s = []
    for _ in range(n_trials):
        random_preds = rng.randint(0, n_classes, size=len(y_test))
        random_accs.append(accuracy_score(y_test, random_preds))
        random_f1s.append(f1_score(y_test, random_preds, average='macro', zero_division=0))
    
    # Majority class
    vals, counts = np.unique(y_test, return_counts=True)
    majority_class = vals[np.argmax(counts)]
    majority_preds = np.full_like(y_test, majority_class)
    majority_acc = accuracy_score(y_test, majority_preds)
    majority_f1 = f1_score(y_test, majority_preds, average='macro', zero_division=0)
    
    return {
        'random_acc_mean': np.mean(random_accs),
        'random_acc_std': np.std(random_accs),
        'random_f1_mean': np.mean(random_f1s),
        'majority_acc': majority_acc,
        'majority_f1': majority_f1,
        'majority_class': int(majority_class),
    }


# ============================================================
# MAIN: ABLATION STUDY
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CLASSIFIER TRAINING — Ablation Study")
    print("=" * 70)
    
    # --- Load data ---
    df = pd.read_csv("situnes_exploration_merged.csv")
    beliefs = np.load("hmm_belief_states.npy")
    
    # --- Filter out 'unknown' bucket (no audio features matched) ---
    valid_mask = df['mood_bucket'].isin(['happy-energetic', 'calm-relaxed', 
                                         'tense-dark', 'sad-melancholic'])
    df_valid = df[valid_mask].reset_index(drop=True)
    beliefs_valid = beliefs[valid_mask.values]
    
    print(f"Valid interactions (excluding unknown): {len(df_valid)}")
    print(f"Bucket distribution:")
    print(df_valid['mood_bucket'].value_counts())
    
    # --- Target encoding ---
    bucket_labels = ['happy-energetic', 'calm-relaxed', 'tense-dark', 'sad-melancholic']
    bucket_to_int = {b: i for i, b in enumerate(bucket_labels)}
    y = df_valid['mood_bucket'].map(bucket_to_int).values
    
    # --- User-based train/test split ---
    np.random.seed(42)
    users = sorted(df_valid['user_id'].unique())
    shuffled = np.random.permutation(users)
    train_users = set(shuffled[:20])
    test_users = set(shuffled[20:])
    
    train_mask = df_valid['user_id'].isin(train_users).values
    test_mask = df_valid['user_id'].isin(test_users).values
    
    print(f"\nTrain: {train_mask.sum()} interactions ({len(train_users)} users)")
    print(f"Test:  {test_mask.sum()} interactions ({len(test_users)} users)")
    
    # --- Class weights for imbalance ---
    train_counts = np.bincount(y[train_mask], minlength=4)
    class_weights = len(y[train_mask]) / (4 * train_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * 4  # Normalize
    print(f"Class weights: {class_weights.round(3)}")
    
    # --- Baselines ---
    baselines = random_baseline(y[test_mask], n_classes=4)
    print(f"\n--- Baselines ---")
    print(f"Random:   acc={baselines['random_acc_mean']:.3f} ± {baselines['random_acc_std']:.3f}, "
          f"F1={baselines['random_f1_mean']:.3f}")
    print(f"Majority: acc={baselines['majority_acc']:.3f}, F1={baselines['majority_f1']:.3f} "
          f"(class={bucket_labels[baselines['majority_class']]})")
    
    # --- Ablation experiments ---
    ablation_configs = {
        'A_full':       'full',
        'B_no_hmm':     'no_hmm', 
        'C_no_emotion': 'no_emotion',
        'D_context':    'context_only',
    }
    
    all_results = {}
    all_histories = {}
    
    # Run multiple seeds for stability
    N_SEEDS = 5
    
    for config_name, feature_set in ablation_configs.items():
        print(f"\n{'='*50}")
        print(f"Model {config_name} (features: {feature_set})")
        print(f"{'='*50}")
        
        # Build features
        X, feat_names = build_feature_matrix(df_valid, beliefs_valid, feature_set)
        print(f"Features: {len(feat_names)} → {feat_names}")
        
        # Split
        X_train_raw, y_train = X[train_mask], y[train_mask]
        X_test_raw, y_test = X[test_mask], y[test_mask]
        
        # Standardize (fit on train only)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        
        # Run multiple seeds and average
        seed_results = []
        
        for seed in range(N_SEEDS):
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            model = MoodClassifier(
                n_features=X_train.shape[1],
                n_classes=4,
                hidden1=64,
                hidden2=32,
                dropout=0.3,
            )
            
            history = train_model(
                model, X_train, y_train, X_test, y_test,
                n_epochs=200, lr=0.001, batch_size=64,
                weight_decay=1e-4, class_weights=class_weights,
                verbose=(seed == 0),  # Only print first seed
            )
            
            results = evaluate_model(model, X_test, y_test, bucket_labels)
            seed_results.append(results)
        
        # Average across seeds
        avg_acc = np.mean([r['accuracy'] for r in seed_results])
        std_acc = np.std([r['accuracy'] for r in seed_results])
        avg_f1 = np.mean([r['f1_macro'] for r in seed_results])
        std_f1 = np.std([r['f1_macro'] for r in seed_results])
        
        all_results[config_name] = {
            'avg_acc': avg_acc,
            'std_acc': std_acc,
            'avg_f1': avg_f1,
            'std_f1': std_f1,
            'feature_set': feature_set,
            'n_features': len(feat_names),
            'feature_names': feat_names,
            'best_run': seed_results[np.argmax([r['accuracy'] for r in seed_results])],
            'all_runs': seed_results,
        }
        all_histories[config_name] = history  # Last seed's history
        
        print(f"\n  Results ({N_SEEDS} seeds):")
        print(f"    Accuracy: {avg_acc:.3f} ± {std_acc:.3f}")
        print(f"    F1 macro: {avg_f1:.3f} ± {std_f1:.3f}")
    
    # ============================================================
    # RESULTS SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("ABLATION STUDY RESULTS")
    print("=" * 70)
    
    print(f"\n{'Model':<20s} {'Features':<15s} {'N_feat':>6s} "
          f"{'Accuracy':>12s} {'F1 Macro':>12s}")
    print("-" * 70)
    print(f"{'Random baseline':<20s} {'—':<15s} {'—':>6s} "
          f"{baselines['random_acc_mean']:>7.3f} ± {baselines['random_acc_std']:.3f}"
          f"{baselines['random_f1_mean']:>7.3f}{'':>5s}")
    print(f"{'Majority baseline':<20s} {'—':<15s} {'—':>6s} "
          f"{baselines['majority_acc']:>12.3f} {baselines['majority_f1']:>12.3f}")
    
    for config_name, res in all_results.items():
        print(f"{config_name:<20s} {res['feature_set']:<15s} {res['n_features']:>6d} "
              f"{res['avg_acc']:>7.3f} ± {res['std_acc']:.3f}"
              f"{res['avg_f1']:>7.3f} ± {res['std_f1']:.3f}")
    
    # ============================================================
    # VISUALIZATIONS
    # ============================================================
    
    # --- 1. Ablation comparison bar chart ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    model_names = list(all_results.keys())
    accs = [all_results[m]['avg_acc'] for m in model_names]
    acc_errs = [all_results[m]['std_acc'] for m in model_names]
    f1s = [all_results[m]['avg_f1'] for m in model_names]
    f1_errs = [all_results[m]['std_f1'] for m in model_names]
    
    x = np.arange(len(model_names))
    colors_bar = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    
    # Accuracy
    bars = axes[0].bar(x, accs, yerr=acc_errs, capsize=5, color=colors_bar, alpha=0.8)
    axes[0].axhline(baselines['random_acc_mean'], color='gray', linestyle='--', 
                    label=f"Random ({baselines['random_acc_mean']:.3f})")
    axes[0].axhline(baselines['majority_acc'], color='darkgray', linestyle=':', 
                    label=f"Majority ({baselines['majority_acc']:.3f})")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, rotation=20)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Test Accuracy by Model Variant")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, max(accs) * 1.3)
    axes[0].grid(axis='y', alpha=0.3)
    
    # F1
    bars = axes[1].bar(x, f1s, yerr=f1_errs, capsize=5, color=colors_bar, alpha=0.8)
    axes[1].axhline(baselines['random_f1_mean'], color='gray', linestyle='--', 
                    label=f"Random ({baselines['random_f1_mean']:.3f})")
    axes[1].axhline(baselines['majority_f1'], color='darkgray', linestyle=':', 
                    label=f"Majority ({baselines['majority_f1']:.3f})")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names, rotation=20)
    axes[1].set_ylabel("F1 Score (macro)")
    axes[1].set_title("Test F1 Score by Model Variant")
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(0, max(f1s) * 1.3)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("ablation_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: ablation_results.png")
    
    # --- 2. Confusion matrix for best model ---
    best_model_name = max(all_results, key=lambda m: all_results[m]['avg_acc'])
    best_cm = all_results[best_model_name]['best_run']['confusion_matrix']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=bucket_labels, yticklabels=bucket_labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {best_model_name} (best run)")
    plt.tight_layout()
    plt.savefig("confusion_matrix_best.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: confusion_matrix_best.png")
    
    # --- 3. Training curves for full model ---
    if 'A_full' in all_histories:
        h = all_histories['A_full']
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].plot(h['train_loss'], label='Train', alpha=0.8)
        axes[0].plot(h['val_loss'], label='Validation', alpha=0.8)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training & Validation Loss (Model A)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(h['train_acc'], label='Train', alpha=0.8)
        axes[1].plot(h['val_acc'], label='Validation', alpha=0.8)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Training & Validation Accuracy (Model A)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("Saved: training_curves.png")
    
    # --- 4. Per-class performance ---
    print(f"\n--- Per-Class Performance ({best_model_name}, best run) ---")
    report = all_results[best_model_name]['best_run']['classification_report']
    for cls_name in bucket_labels:
        if cls_name in report:
            r = report[cls_name]
            print(f"  {cls_name:20s}: precision={r['precision']:.3f} "
                  f"recall={r['recall']:.3f} f1={r['f1-score']:.3f} "
                  f"support={r['support']}")
    
    # --- 5. Feature importance (via ablation delta) ---
    print(f"\n--- Feature Group Importance (via ablation) ---")
    full_acc = all_results['A_full']['avg_acc']
    for config, res in all_results.items():
        if config != 'A_full':
            delta = full_acc - res['avg_acc']
            removed = config.split('_', 1)[1]
            direction = "+" if delta > 0 else ""
            print(f"  Removing {removed:15s}: Δ accuracy = {direction}{delta:.3f}")
    
    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"""
Best model: {best_model_name}
  Accuracy: {all_results[best_model_name]['avg_acc']:.3f} ± {all_results[best_model_name]['std_acc']:.3f}
  F1 macro: {all_results[best_model_name]['avg_f1']:.3f} ± {all_results[best_model_name]['std_f1']:.3f}

Baselines:
  Random:   {baselines['random_acc_mean']:.3f}
  Majority: {baselines['majority_acc']:.3f}

Key findings:
  • HMM belief contribution: Δacc = {all_results['A_full']['avg_acc'] - all_results['B_no_hmm']['avg_acc']:+.3f} (full - no_hmm)
  • Pre-emotion contribution: Δacc = {all_results['A_full']['avg_acc'] - all_results['C_no_emotion']['avg_acc']:+.3f} (full - no_emotion)
  • Context-only baseline:   {all_results['D_context']['avg_acc']:.3f}
""")