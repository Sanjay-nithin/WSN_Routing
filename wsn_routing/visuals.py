import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report

# Ensure output directory exists
os.makedirs('outputs', exist_ok=True)

# Set custom styling for premium looks
sns.set_theme(style='whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'font.family': 'sans-serif'
})

def plot_training_history(history, save_path='outputs/cnn_training_curves.png'):
    """
    Plots the training and testing loss and accuracy curves side-by-side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Extract data
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Loss Curve
    axes[0].plot(epochs, history['train_loss'], label='Train Loss', color='#3b82f6', linewidth=2.5, marker='o', markersize=4)
    axes[0].plot(epochs, history['test_loss'], label='Test Loss', color='#ef4444', linewidth=2.5, marker='s', markersize=4)
    axes[0].set_title('Training & Test Loss', pad=15)
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend(frameon=True, facecolor='white', edgecolor='none')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # 2. Accuracy Curve
    train_acc_pct = [acc * 100 for acc in history['train_acc']]
    test_acc_pct = [acc * 100 for acc in history['test_acc']]
    axes[1].plot(epochs, train_acc_pct, label='Train Acc', color='#10b981', linewidth=2.5, marker='o', markersize=4)
    axes[1].plot(epochs, test_acc_pct, label='Test Acc', color='#8b5cf6', linewidth=2.5, marker='s', markersize=4)
    axes[1].set_title('Training & Test Accuracy', pad=15)
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend(frameon=True, facecolor='white', edgecolor='none')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"[Visuals] Saved training history curves to {save_path}")

def plot_confusion_matrix(y_true, y_pred, class_names=['Healthy', 'Congested', 'Unhealthy'], save_path='outputs/cnn_confusion_matrix.png'):
    """
    Computes and plots a beautiful, normalized confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # normalized
    
    plt.figure(figsize=(7, 6))
    
    # Create text annotations with both raw counts and percentages
    labels = (np.array([f"{val}\n({pct:.1%})" for val, pct in zip(cm.flatten(), cm_norm.flatten())])).reshape(cm.shape)
    
    # Custom color palette (sleek blues)
    sns.heatmap(
        cm_norm * 100, 
        annot=labels, 
        fmt="", 
        cmap='crest', 
        xticklabels=class_names, 
        yticklabels=class_names,
        cbar=True,
        cbar_kws={'label': 'Classification Rate (%)'},
        linewidths=1.5,
        linecolor='white',
        square=True
    )
    
    plt.title('Normalized Confusion Matrix', pad=20)
    plt.ylabel('True Class', labelpad=10)
    plt.xlabel('Predicted Class', labelpad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"[Visuals] Saved confusion matrix to {save_path}")

def print_and_plot_report(y_true, y_pred, class_names=['Healthy', 'Congested', 'Unhealthy'], save_path='outputs/cnn_per_class_metrics.png'):
    """
    Prints the classification report and plots the precision, recall, and F1-score for each class.
    """
    # Print textual report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    print("\n" + "="*50)
    print("               CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=class_names))
    print("="*50)
    
    # Extract per-class metrics for plotting
    classes = class_names
    metrics = ['precision', 'recall', 'f1-score']
    
    data = []
    for cls in classes:
        for metric in metrics:
            data.append({
                'Class': cls,
                'Metric': metric.capitalize(),
                'Value': report[cls][metric] * 100
            })
            
    import pandas as pd
    df_metrics = pd.DataFrame(data)
    
    plt.figure(figsize=(9, 5))
    
    # Palette definition
    palette = {'Precision': '#2563eb', 'Recall': '#d97706', 'F1-score': '#7c3aed'}
    
    ax = sns.barplot(
        data=df_metrics, 
        x='Class', 
        y='Value', 
        hue='Metric', 
        palette=palette,
        edgecolor='black',
        linewidth=0.5
    )
    
    # Add values on top of bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}%',
                        xy=(p.get_x() + p.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
            
    plt.title('Performance Metrics by Node Status Class', pad=15)
    plt.ylabel('Percentage (%)')
    plt.xlabel('Node Status')
    plt.ylim(0, 110)
    plt.legend(frameon=True, facecolor='white', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"[Visuals] Saved class metrics plot to {save_path}")
