import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import optuna
import pandas as pd
from dataset import get_raw_splits, get_loaders, HemorrhageDataset, get_transforms
from models import get_pretrained_model, CustomCNN
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings

# Suppress the common PIL palette transparency warning
warnings.filterwarnings("ignore", "(?i)Palette images with Transparency")

# Focal Loss Implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.15):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha # Class weights tensor
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha, label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return torch.mean(focal_loss)

def train_one_epoch(model, loader, criterion, optimizer, device, desc="Training"):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    pbar = tqdm(loader, desc=desc, leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    return running_loss / len(loader.dataset), running_corrects.double() / len(loader.dataset)

def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)
    return val_loss / len(loader.dataset), val_corrects.double() / len(loader.dataset)

def objective(trial, model_type, train_val_df):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == "CustomCNN":
       lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    else:
        lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    fold_accs = []
    
    # 1 Fold is enough for quick tuning exploration
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_df, train_val_df['hemorrhage'])):
        train_df = train_val_df.iloc[train_idx]
        val_df = train_val_df.iloc[val_idx]
        
        train_loader, val_loader = get_loaders(train_df, val_df, batch_size=32)
        
        if model_type == "ConvNext":
            model = get_pretrained_model(fine_tune=True)
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            model = CustomCNN()
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        model.to(device)
        
        y_t = train_df['hemorrhage'].values
        cw = compute_class_weight('balanced', classes=np.unique(y_t), y=y_t)
        wt = torch.tensor(cw, dtype=torch.float32).to(device)
        criterion = FocalLoss(alpha=wt, gamma=3.0)
        
        best_v_acc = 0
        patience = 4
        trigger = 0
        
        for epoch in range(3): # Enough epochs to see early trajectory
            _, _ = train_one_epoch(model, train_loader, criterion, optimizer, device, desc=f"Trial {trial.number} Ep {epoch+1}")
            _, v_acc = validate(model, val_loader, criterion, device)
            if v_acc > best_v_acc:
                best_v_acc = v_acc
                trigger = 0
            else:
                trigger += 1
                if trigger >= 3: break # Reduced tuning patience
        
        fold_accs.append(best_v_acc.item())
        break # Only compute 1st fold for tuning speed
        
    return np.mean(fold_accs)

def final_evaluate(model, loader, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    p = precision_score(all_labels, all_preds, zero_division=0)
    r = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0
    cm = confusion_matrix(all_labels, all_preds)
    
    with open(f"output/{model_name}_metrics.txt", "w") as f:
        f.write(f"Metrics (Independent Test Set)\nAccuracy: {acc:.4f}\nPrecision: {p:.4f}\nRecall: {r:.4f}\nF1: {f1:.4f}\nAUC: {auc:.4f}\n")
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Hemo'], yticklabels=['Normal', 'Hemo'])
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(f'output/{model_name}_cm.png'); plt.close()

def generate_final_comparisons(all_results):
    """Generates ROC comparison and Metric comparison plots for all trained models."""
    print("\nGenerating final comparison plots...")
    
    # 1. ROC Curves Comparison
    plt.figure(figsize=(10, 8))
    for res in all_results:
        fpr, tpr, _ = roc_curve(res['labels'], res['probs'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{res['name']} (AUC = {roc_auc:.4f})")
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.savefig("output/roc_curves_comparison.png"); plt.close()

    # 2. Metrics Bar Chart Comparison
    metrics_data = []
    for res in all_results:
        metrics_data.append({
            "Model": res['name'],
            "Accuracy": accuracy_score(res['labels'], res['preds']),
            "Precision": precision_score(res['labels'], res['preds'], zero_division=0),
            "Recall": recall_score(res['labels'], res['preds'], zero_division=0),
            "F1-Score": f1_score(res['labels'], res['preds'], zero_division=0),
            "AUC": roc_auc_score(res['labels'], res['probs']) if len(np.unique(res['labels'])) > 1 else 0
        })
    
    df = pd.DataFrame(metrics_data)
    df_melted = df.melt(id_vars="Model", var_name="Metric", value_name="Value")
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_melted, x="Metric", y="Value", hue="Model", palette="viridis")
    plt.title("Model Performance Comparison")
    plt.ylim(0, 1.1); plt.grid(axis='y', alpha=0.3)
    plt.savefig("output/model_comparison.png"); plt.close()
    print("Comparison plots saved to output/ folder.")

def run_full_pipeline():
    os.makedirs("output", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading datasets (including external samples)...")
    train_val_df, test_df = get_raw_splits()
    
    all_eval_results = []

    for m_type in ["CustomCNN", "ConvNext"]:
        print(f"\n{'='*40}")
        print(f"--- Model: {m_type} ---")
        print(f"{'='*40}")
        
        # Hyperparameter Tuning
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: objective(t, m_type, train_val_df), n_trials=5)
        
        # Save Tuning Results
        df_trials = study.trials_dataframe()
        df_trials.to_csv(f"output/{m_type}_tuning_results.csv", index=False)
        
        best_lr = study.best_params['lr']
        best_wd = study.best_params['weight_decay']
        print(f"Best Trial params: LR={best_lr:.6f}, WD={best_wd:.6f}. Training final model...")
        
        train_df, val_df = train_test_split(train_val_df, test_size=0.15, stratify=train_val_df['hemorrhage'], random_state=42)
        train_loader, val_loader = get_loaders(train_df, val_df, batch_size=16)
        
        y_train = train_df['hemorrhage'].values
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

        if m_type == "ConvNext":
            model = get_pretrained_model(fine_tune=True)
            optimizer = optim.AdamW(model.parameters(), lr=best_lr, weight_decay=best_wd)
        else:
            model = CustomCNN()
            optimizer = optim.AdamW(model.parameters(), lr=best_lr, weight_decay=best_wd)
            
        model.to(device)
        criterion = FocalLoss(alpha=weights_tensor, gamma=3.0)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        
        best_v_acc = 0
        history = {'t_loss': [], 'v_loss': [], 't_acc': [], 'v_acc': []}
        patience = 5; trigger = 0
        
        for epoch in range(40):
            t_l, t_a = train_one_epoch(model, train_loader, criterion, optimizer, device, desc=f"Epoch {epoch+1}")
            v_l, v_a = validate(model, val_loader, criterion, device)
            scheduler.step(v_l)
            history['t_loss'].append(t_l); history['v_loss'].append(v_l)
            history['t_acc'].append(t_a.item()); history['v_acc'].append(v_a.item())
            
            if v_a > best_v_acc:
                best_v_acc = v_a
                trigger = 0
                torch.save(model.state_dict(), f"output/{m_type.lower()}_hemorrhage.pth")
            else:
                trigger += 1
            
            print(f"Epoch {epoch+1}/50 | T-Loss: {t_l:.4f} | T-Acc: {t_a:.4f} | V-Loss: {v_l:.4f} | V-Acc: {v_a:.4f} | Best: {best_v_acc:.4f}")
            if trigger >= patience: break
                
        # Final Evaluation on Test Set
        model.load_state_dict(torch.load(f"output/{m_type.lower()}_hemorrhage.pth", map_location=device, weights_only=True))
        _, v_trans = get_transforms()
        test_ds = HemorrhageDataset(test_df, v_trans)
        test_loader = test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
        
        # Capture raw results for final comparison
        model.eval()
        p_probs, p_labels, p_preds = [], [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                p_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
                p_labels.extend(labels.cpu().numpy())
                p_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        
        all_eval_results.append({
            "name": m_type, "probs": p_probs, "labels": p_labels, "preds": p_preds
        })
        
        # Individual model plots
        final_evaluate(model, test_loader, m_type)
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1); plt.plot(history['t_loss'], label='Train'); plt.plot(history['v_loss'], label='Val'); plt.legend(); plt.title(f"{m_type} Loss")
        plt.subplot(1, 2, 2); plt.plot(history['t_acc'], label='Train'); plt.plot(history['v_acc'], label='Val'); plt.legend(); plt.title(f"{m_type} Accuracy")
        plt.savefig(f'output/{m_type}_training.png'); plt.close()
        print(f"Completed {m_type}. Models and plots saved in output/ folder.")

    # GENERATE FINAL COMPARISONS
    generate_final_comparisons(all_eval_results)

if __name__ == "__main__":
    run_full_pipeline()
