import matplotlib.pyplot as plt
import pandas as pd
import os
import textwrap

def generate_optimization_strategies_table():
    strategies = {
        "Optimization Strategy": [
            "Hyperparameter Tuning",
            "Advanced Loss Function",
            "Transfer Learning",
            "Data Augmentation (5x)",
            "LR Scheduling",
            "Regularization Tech",
            "Class Imbalance Handling",
            "Early Stopping",
            "Optimizer Choice",
            "Fine-Tuning Strategy"
        ],
        "Method / Technique": [
            "Optuna Bayesian Optimization",
            "Weighted Focal Loss (γ=3.0)",
            "ConvNext Tiny (Pretrained)",
            "Random Flip, Rotate, Crop, Jitter",
            "ReduceLROnPlateau",
            "Dropout(0.6), Label Smoothing(0.15)",
            "Dynamic Class Weights (Balanced)",
            "5-Epoch Patience Monitoring",
            "AdamW (Decoupled Weight Decay)",
            "Selective Layer Unfreezing"
        ],
        "Objective / Benefit": [
            "Finds optimal LR and WD automatically via Bayesian search",
            "Focuses learning on 'hard' misclassified samples while ignoring easy ones",
            "Leverages pre-learned visual features from 1.2M ImageNet images",
            "Prevents overfitting via 30K+ image variants with geometric/color noise",
            "Fine-tunes convergence in later stages by reducing step size when stalled",
            "Generalizes better, prevents overconfidence and helps model stay robust",
            "Ensures fairness between normal/hemo classes using penalty adjustment",
            "Prevents training on noise; saves time and hardware resources",
            "More stable weight updates vs standard Adam through decoupled decay",
            "Adapts generic textures to specific medical domain patterns"
        ]
    }

    # Apply text wrapping to longer descriptions
    for col in strategies:
        strategies[col] = [textwrap.fill(text, width=40) for text in strategies[col]]

    df = pd.DataFrame(strategies)

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.axis('off')
    ax.axis('tight')

    # Color palette
    header_color = '#00695c' 
    row_colors = ['#ffffff', '#e0f2f1']
    edge_color = '#b2dfdb'

    # Create table
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     cellLoc='left',
                     loc='center',
                     colColours=[header_color]*len(df.columns))

    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 3.5) # Vertical scale for multi-line text

    # Column widths
    table.auto_set_column_width(col=[0, 1, 2])

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(edge_color)
        if row == 0:
            cell.set_text_props(weight='bold', color='white', fontsize=12, ha='center')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[row % len(row_colors)])
            if col == 0:
                cell.set_text_props(weight='bold', color='#004d40')
            cell.set_text_props(va='center')

    plt.title("Model Optimization & Performance Maximization Strategies", 
              fontsize=22, weight='bold', pad=60, color='#004d40')
    
    # Save directory
    os.makedirs("output", exist_ok=True)
    output_path = "output/optimization_strategies.png"
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=False)
    plt.close()
    print(f"Fixed optimization strategies table saved to {output_path}")

if __name__ == "__main__":
    generate_optimization_strategies_table()
