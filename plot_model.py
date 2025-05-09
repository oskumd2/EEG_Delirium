import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_score, recall_score, auc, RocCurveDisplay
import seaborn
warnings.filterwarnings("ignore")

def brier_score_loss(y_true, y_pred):
    """
    Calculate Brier score loss between true labels and predicted probabilities
    Lower values indicate better calibrated predictions (0 is perfect)
    """
    return np.mean((y_true - y_pred) ** 2)

def integrated_calibration_index(y_true, y_pred, n_bins=100):
    """
    Calculate integrated calibration index (ICI)
    Lower values indicate better calibration (0 is perfect)
    """
    # Sort predictions and corresponding true values
    sort_idx = np.argsort(y_pred)
    y_pred_sorted = y_pred[sort_idx]
    y_true_sorted = y_true[sort_idx]
    
    # Calculate calibration curve using moving average
    window = len(y_true) // n_bins
    if window < 1:
        window = 1
    
    calibration_curve = np.array([
        np.mean(y_true_sorted[max(0, i-window):min(len(y_true), i+window)])
        for i in range(len(y_true))
    ])
    
    # Calculate absolute difference between predictions and calibration curve
    ici = np.mean(np.abs(y_pred_sorted - calibration_curve))
    return ici

def bootstrap_ci(y_true, y_pred, y_pred_binary, n_bootstraps=4000, alpha=0.05):
    """Calculate bootstrap confidence intervals for various metrics."""
    
    n_samples = len(y_true)
    results = {
        'AUROC': [],
        'AUPRC': [],
        'F1': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'Brier': [],
        'ICI': []
    }
    
    for _ in range(n_bootstraps):
        indices = np.random.choice(n_samples, n_samples, replace=True)
            
        # Get bootstrap samples
        boot_y_true = y_true[indices]
        boot_y_pred = y_pred[indices]
        boot_y_pred_binary = y_pred_binary[indices]
        
        # Calculate metrics
        results['AUROC'].append(roc_auc_score(boot_y_true, boot_y_pred))
        results['AUPRC'].append(average_precision_score(boot_y_true, boot_y_pred))
        results['F1'].append(f1_score(boot_y_true, boot_y_pred_binary))
        results['Accuracy'].append(accuracy_score(boot_y_true, boot_y_pred_binary))
        results['Precision'].append(precision_score(boot_y_true, boot_y_pred_binary))
        results['Recall'].append(recall_score(boot_y_true, boot_y_pred_binary))
        results['Brier'].append(brier_score_loss(boot_y_true, boot_y_pred))
        results['ICI'].append(integrated_calibration_index(boot_y_true, boot_y_pred))
    
    # Calculate confidence intervals
    ci = {}
    for metric, values in results.items():
        # Filter out any NaN values that might have occurred
        values = np.array(values)
        values = values[~np.isnan(values)]
        if len(values) > 0:
            lower = np.percentile(values, alpha/2 * 100)
            upper = np.percentile(values, (1 - alpha/2) * 100)
            ci[metric] = (lower, upper)
        else:
            ci[metric] = (np.nan, np.nan)
    
    return ci

def draw_model_evaluation_plots(y_test, y_test_proba, y_pred_test, n_bins=10):
    AUROC = roc_auc_score(y_test, y_test_proba)
    AUPRC = average_precision_score(y_test, y_test_proba)
    F1_score = f1_score(y_test, y_pred_test)
    brier_score = brier_score_loss(y_test, y_test_proba)
    ici = integrated_calibration_index(y_test, y_test_proba)
    ci_metrics = bootstrap_ci(y_test, y_test_proba, y_pred_test)
    print(f'AUROC {AUROC:.3f} (95% CI: {ci_metrics["AUROC"][0]:.3f}-{ci_metrics["AUROC"][1]:.3f})')
    print(f'AUPRC {AUPRC:.3f} (95% CI: {ci_metrics["AUPRC"][0]:.3f}-{ci_metrics["AUPRC"][1]:.3f})')
    print(f'F1 Score {F1_score:.3f} (95% CI: {ci_metrics["F1"][0]:.3f}-{ci_metrics["F1"][1]:.3f})')
    print(f'Test Accuracy {(accuracy_score(y_test, y_pred_test)):.3f} (95% CI: {ci_metrics["Accuracy"][0]:.3f}-{ci_metrics["Accuracy"][1]:.3f})')
    print(f'Brier Score {brier_score:.3f} (95% CI: {ci_metrics["Brier"][0]:.3f}-{ci_metrics["Brier"][1]:.3f})')
    print(f'ICI {ici:.3f} (95% CI: {ci_metrics["ICI"][0]:.3f}-{ci_metrics["ICI"][1]:.3f})')
    """
    Draw four evaluation plots in a 1x4 layout:
    1. Calibration plot with error bars
    2. ROC curve
    3. Confusion matrix
    4. Probability histogram
    """
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), tight_layout=True)
    
    # Ensure all plots have the same height and size
    for ax in axes:
        ax.set_box_aspect(1)
    
    # 1. Calibration plot with error bars
    plt.sca(axes[0])
    # Add (a) label above the plot
    axes[0].set_title('(a)', fontsize=14, fontweight='bold', loc='left', pad=10)
    
    # Create bins and find probabilities in each bin
    bins = np.linspace(0, 1, n_bins + 1)
    binids = np.digitize(y_test_proba, bins) - 1
    
    bin_true = np.zeros(n_bins)
    bin_pred = np.zeros(n_bins)
    bin_errors = np.zeros(n_bins)
    bin_sizes = np.zeros(n_bins)
    
    # Calculate statistics for each bin
    for bin_idx in range(n_bins):
        bin_mask = binids == bin_idx
        if np.sum(bin_mask) > 0:  # Check if bin has any samples
            bin_sizes[bin_idx] = np.sum(bin_mask)
            bin_true[bin_idx] = np.mean(y_test[bin_mask])
            bin_pred[bin_idx] = np.mean(y_test_proba[bin_mask])
            # Calculate standard error for the bin
            bin_errors[bin_idx] = np.std(y_test[bin_mask]) / np.sqrt(bin_sizes[bin_idx])
    
    # Plot calibration
    plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
    plt.errorbar(bin_pred, bin_true, yerr=bin_errors, 
                fmt='o', color='red', ecolor='gray', 
                capsize=3, capthick=1, markersize=4,
                label='Model calibration')
    
    plt.xlabel('Mean predicted probability')
    plt.ylabel('True probability')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 2. ROC curve
    plt.sca(axes[1])
    # Add (b) label above the plot
    axes[1].set_title('(b)', fontsize=14, fontweight='bold', loc='left', pad=10)
    
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k:', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # 3. Confusion matrix
    plt.sca(axes[2])
    # Add (c) label above the plot
    axes[2].set_title('(c)', fontsize=14, fontweight='bold', loc='left', pad=10)
    
    fontsize = 12
    contingency_table = pd.crosstab(y_test, y_pred_test)
    contingency_table.columns = ['Normal', 'Delirium']
    contingency_table.columns.name = 'Label'
    contingency_table.index = ['Normal', 'Delirium']
    contingency_table.index.name = 'Prediction'
    contingency_percentage = contingency_table / len(y_test) * 100
    
    ax = seaborn.heatmap(contingency_table, annot=False, fmt="d", cmap='Blues', annot_kws={"size": 12}, ax=axes[2])
    for i in range(contingency_table.shape[0]):
        for j in range(contingency_table.shape[1]):
            if i == 0 and j == 0:
                color = 'white'
            else:
                color = 'black'
            
            count = contingency_table.iloc[i, j]
            percent = contingency_percentage.iloc[i, j]
            text = f"{count:,}\n({percent:.1f}%)"
            ax.text(j+0.5, i+0.5, text, 
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=fontsize, color=color)
    ax.set_xlabel('Prediction', fontsize=fontsize)
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_ticks_position('top')
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize)
    plt.ylabel('Label', fontsize=fontsize)
    
    # 4. Probability histogram
    plt.sca(axes[3])
    # Add (d) label above the plot
    axes[3].set_title('(d)', fontsize=14, fontweight='bold', loc='left', pad=10)
    
    plt.hist(y_test_proba, bins=50, edgecolor='black')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    
    # Align all title positions to ensure (a),(b),(c),(d) are at the same height
    title_y_pos = 1.05
    for ax in axes:
        ax.title.set_position([0.0, title_y_pos])
    
    plt.show()