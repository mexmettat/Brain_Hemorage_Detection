import matplotlib.pyplot as plt
import pandas as pd
import os

def generate_hyperparameter_trials_table():
    # Load raw trial data from CSVs
    try:
        df_conv = pd.read_csv("output/ConvNext_tuning_results.csv")
        df_cnn = pd.read_csv("output/CustomCNN_tuning_results.csv")
    except Exception as e:
        print(f"Error reading CSVs: {e}. Using fallback data.")
        # Fallback in case CSVs are missing or malformed
        return

    # Process ConvNext Trials
    conv_trials = []
    for idx, row in df_conv.iterrows():
        conv_trials.append({
            "Model": "ConvNext Tiny",
            "Trial": f"#{int(row['number'])}",
            "Learning Rate": f"{row['params_lr']:.2e}",
            "Weight Decay": f"{row['params_weight_decay']:.2e}",
            "Batch Size": "32 (Tune) / 16 (Final)",
            "Max Epochs": "40 (Final)",
            "Patience": "5",
            "Val Acc": f"{row['value']:.4f}"
        })

    # Process CustomCNN Trials
    cnn_trials = []
    for idx, row in df_cnn.iterrows():
        cnn_trials.append({
            "Model": "Custom CNN",
            "Trial": f"#{int(row['number'])}",
            "Learning Rate": f"{row['params_lr']:.2e}",
            "Weight Decay": f"{row['params_weight_decay']:.2e}",
            "Batch Size": "32 (Tune) / 16 (Final)",
            "Max Epochs": "40 (Final)",
            "Patience": "5",
            "Val Acc": f"{row['value']:.4f}"
        })

    # Combine all trials
    all_trials = conv_trials + cnn_trials
    df = pd.DataFrame(all_trials)

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.axis('off')
    ax.axis('tight')

    # Color palette - Professional Blue Theme
    header_color = '#0d47a1' 
    row_colors = ['#ffffff', '#e3f2fd']
    edge_color = '#bbdefb'

    # Create table
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center',
                     colColours=[header_color]*len(df.columns))

    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 3.0)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(edge_color)
        if row == 0:
            cell.set_text_props(weight='bold', color='white', fontsize=12)
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[row % len(row_colors)])
            # Highlight Best Trials
            val_acc_float = float(df.iloc[row-1]['Val Acc'])
            # Highlight top performance for each model category
            if (df.iloc[row-1]['Model'] == "ConvNext Tiny" and val_acc_float > 0.974) or \
               (df.iloc[row-1]['Model'] == "Custom CNN" and val_acc_float > 0.810):
                cell.set_facecolor('#c8e6c9') # Distinct green for winners
                cell.set_text_props(weight='bold')

    plt.title("Hyperparameter Optimization: Experimental Trials & Selection", 
              fontsize=22, weight='bold', pad=60, color='#0d47a1')
    
    # Adding a footnote for context
    plt.figtext(0.5, 0.05, "* Trials #0-4 represent the Optuna Bayesian Search phase. Winners were selected for the Final Training (40 Epochs, Patience 5).", 
               ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.1, "pad":5})

    # Save directory
    os.makedirs("output", exist_ok=True)
    output_path = "output/hyperparameter_trials.png"
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=False)
    plt.close()
    print(f"Hyperparameter trials table saved to {output_path}")

if __name__ == "__main__":
    generate_hyperparameter_trials_table()
