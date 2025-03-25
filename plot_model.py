import os, shutil, random, time, base64, pickle
from collections import Counter
from glob import glob
from tqdm import tqdm
from PIL import Image
from lxml import etree as ET
#import xml.etree.ElementTree as ET
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import resample, shuffle
from sklearn.metrics import accuracy_score, roc_auc_score,RocCurveDisplay, roc_curve, auc
from scipy import signal
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.model_selection import GroupKFold
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.utils import resample
import seaborn

def draw_roc_curve(y_test, y_test_proba):
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                            estimator_name='Model')
    display.plot()
    plt.show()    

def draw_confusion_matrix(y_test, y_pred_test):
    fontsize = 20
    contingency_table = pd.crosstab(y_test, y_pred_test)
    contingency_table
    contingency_table.columns = ['Normal', 'Delirium']
    contingency_table.columns.name = 'Label'
    contingency_table.index = ['Normal', 'Delirium']
    contingency_table.index.name = 'Prediction'
    contingency_percentage = contingency_table / len(y_test) * 100
    plt.figure(figsize=(8, 6))
    ax = seaborn.heatmap(contingency_table, annot=False, fmt="d", cmap='Blues', annot_kws={"size": 16})
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
    plt.show()

def draw_y_test_proba(y_test_proba):
    plt.hist(y_test_proba, bins=50, edgecolor='black')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.title('Test probability histogram')
    plt.show()

def draw_calibration_plot(prob_true, prob_pred , y_test_proba, n_bins, ax_end):
    hist, bin_edges = np.histogram(prob_true, bins=n_bins, range=(0, 1))
    bin_std = []
    for i in range(n_bins):
        bin_data = y_test_proba[(y_test_proba >= bin_edges[i]) & (y_test_proba < bin_edges[i+1])]
        bin_std.append(np.std(bin_data))
    
    bin_std = [value for value in bin_std if not np.isnan(value)]
    fig, ax = plt.subplots()
    ax.errorbar(prob_pred, prob_true , fmt='o', color='black', yerr=bin_std, markersize=3) #yerr=bin_std
    ax.plot([0, ax_end], [0, ax_end], "k:", label="Perfectly calibrated")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("True probability")
    ax.set_title("Calibration Curve")
    ax.legend()
    plt.show()


def draw_calibration_plot_with_error_bars(y_test, y_test_proba, n_bins=10):
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
    
    # Create the plot
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
    
    # Plot points with error bars
    plt.errorbar(bin_pred, bin_true, yerr=bin_errors, 
                fmt='o', color='red', ecolor='gray', 
                capsize=3, capthick=1, markersize=4,
                label='Model calibration')
    
    plt.xlabel('Mean predicted probability')
    plt.ylabel('True probability')
    #plt.title('Calibration Plot with Error Bars')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.show()

def draw_scatter_plot(y1,y2):
    plt.figure(figsize=(8, 6))
    plt.scatter(y1,y2, alpha=0.5)
    plt.xlabel('Suppression ratio')
    plt.ylabel('Predicted probability of the model')
    plt.grid(True, linestyle='--', alpha=0.7)
    # Add correlation coefficient
    correlation = np.corrcoef(y1,y2)[0, 1]
    plt.annotate(f'Correlation: {correlation:.4f}', 
                 xy=(0.05, 0.95), 
                 xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    plt.tight_layout()
    plt.show()

 
    
def bootstrap_delong_test(y_true, pred1, pred2, n_bootstraps=1000, random_state=42):
    """
    Perform bootstrapped Delong's test to compare two ROC curves.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    pred1 : array-like
        Predictions from first model
    pred2 : array-like
        Predictions from second model
    n_bootstraps : int
        Number of bootstrap samples
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    p_value : float
        p-value from the bootstrapped test
    """
    np.random.seed(random_state)
    n_samples = len(y_true)
    
    # Calculate the original AUC difference
    auc1 = roc_auc_score(y_true, pred1)
    auc2 = roc_auc_score(y_true, pred2)
    observed_diff = abs(auc1 - auc2)
    
    # Bootstrap to get p-value
    count = 0
    for i in range(n_bootstraps):
        # Generate bootstrap sample indices
        indices = resample(range(n_samples), replace=True, n_samples=n_samples)
        
        # Calculate AUCs on bootstrap sample
        if len(np.unique(y_true[indices])) < 2:
            # Skip iteration if bootstrap sample has only one class
            continue
            
        boot_auc1 = roc_auc_score(y_true[indices], pred1[indices])
        boot_auc2 = roc_auc_score(y_true[indices], pred2[indices])
        boot_diff = abs(boot_auc1 - boot_auc2)
        
        # Count how many bootstrap differences are >= observed difference
        if boot_diff >= observed_diff:
            count += 1
    
    # Calculate p-value
    p_value = count / n_bootstraps
    return p_value