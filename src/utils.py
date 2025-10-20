"""
Utility functions for Patient Appointment Prediction project.

This module contains helper functions for metrics calculation,
plotting, and other common operations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)


def calculate_metrics(y_true, y_pred):
    """
    Calculate comprehensive performance metrics.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        
    Returns:
        dict: Dictionary containing all metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=1),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }
    
    # Calculate specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['sensitivity'] = metrics['recall']  # Same as recall
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name="Model", figsize=(8, 6)):
    """
    Plot confusion matrix with annotations.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        model_name (str): Name of the model for title
        figsize (tuple): Figure size
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Show', 'Show'], 
                yticklabels=['No Show', 'Show'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_pred, model_name="Model", figsize=(8, 6)):
    """
    Plot ROC curve for a single model.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        model_name (str): Name of the model for title
        figsize (tuple): Figure size
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_model_comparison(results_dict, metric='f1_score', figsize=(10, 6)):
    """
    Plot comparison of models for a specific metric.
    
    Args:
        results_dict (dict): Dictionary with model names as keys and metrics as values
        metric (str): Metric to compare
        figsize (tuple): Figure size
    """
    models = list(results_dict.keys())
    scores = [results_dict[model][metric] for model in models]
    
    plt.figure(figsize=figsize)
    bars = plt.bar(models, scores, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.title(f'Model Comparison - {metric.replace("_", " ").title()}')
    plt.ylabel(metric.replace("_", " ").title())
    plt.xlabel('Models')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


def plot_feature_importance(importances, feature_names, model_name="Model", top_n=20, figsize=(10, 8)):
    """
    Plot feature importance from tree-based models.
    
    Args:
        importances (array-like): Feature importances
        feature_names (list): Names of features
        model_name (str): Name of the model
        top_n (int): Number of top features to show
        figsize (tuple): Figure size
    """
    # Sort features by importance
    sorted_idx = np.argsort(importances)[::-1][:top_n]
    sorted_importances = importances[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]
    
    plt.figure(figsize=figsize)
    plt.barh(range(len(sorted_names)), sorted_importances, color='royalblue', alpha=0.7)
    plt.yticks(range(len(sorted_names)), sorted_names)
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Feature Importance - {model_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def plot_data_distribution(df, column, title=None, figsize=(8, 6)):
    """
    Plot distribution of a column in the dataset.
    
    Args:
        df (pd.DataFrame): Dataset
        column (str): Column name to plot
        title (str): Plot title
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    
    if df[column].dtype in ['object', 'category']:
        # Categorical data - bar plot
        value_counts = df[column].value_counts()
        plt.bar(range(len(value_counts)), value_counts.values, color='skyblue', alpha=0.7)
        plt.xticks(range(len(value_counts)), value_counts.index, rotation=45)
        plt.ylabel('Count')
    else:
        # Numerical data - histogram
        plt.hist(df[column].dropna(), bins=30, color='skyblue', alpha=0.7, edgecolor='black')
        plt.ylabel('Frequency')
    
    plt.xlabel(column)
    plt.title(title or f'Distribution of {column}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df, title="Correlation Matrix", figsize=(10, 8)):
    """
    Plot correlation heatmap for numerical columns.
    
    Args:
        df (pd.DataFrame): Dataset
        title (str): Plot title
        figsize (tuple): Figure size
    """
    # Select only numerical columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        print("No numerical columns found for correlation analysis.")
        return
    
    correlation_matrix = numeric_df.corr()
    
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                fmt=".2f", vmin=-1, vmax=1, center=0)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def print_classification_report(y_true, y_pred, model_name="Model"):
    """
    Print detailed classification report.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        model_name (str): Name of the model
    """
    print(f"\nClassification Report - {model_name}")
    print("=" * 50)
    print(classification_report(y_true, y_pred, 
                              target_names=['No Show', 'Show']))


def save_results_to_csv(results_dict, filename="model_results.csv"):
    """
    Save model results to CSV file.
    
    Args:
        results_dict (dict): Dictionary with model results
        filename (str): Output filename
    """
    # Convert results to DataFrame
    df_results = pd.DataFrame(results_dict).T
    df_results.to_csv(filename)
    print(f"Results saved to {filename}")


def load_results_from_csv(filename="model_results.csv"):
    """
    Load model results from CSV file.
    
    Args:
        filename (str): Input filename
        
    Returns:
        pd.DataFrame: Results DataFrame
    """
    df_results = pd.read_csv(filename, index_col=0)
    print(f"Results loaded from {filename}")
    return df_results


def create_summary_table(results_dict):
    """
    Create a formatted summary table of model results.
    
    Args:
        results_dict (dict): Dictionary with model results
        
    Returns:
        pd.DataFrame: Formatted results table
    """
    df_results = pd.DataFrame(results_dict).T
    
    # Round numerical columns to 4 decimal places
    numerical_cols = ['accuracy', 'precision', 'recall', 'specificity', 'f1_score']
    for col in numerical_cols:
        if col in df_results.columns:
            df_results[col] = df_results[col].round(4)
    
    return df_results


def plot_learning_curves(model, X, y, cv=5, train_sizes=None, figsize=(10, 6)):
    """
    Plot learning curves for a model.
    
    Args:
        model: Scikit-learn model
        X (array-like): Features
        y (array-like): Target
        cv (int): Number of cross-validation folds
        train_sizes (array-like): Training set sizes to evaluate
        figsize (tuple): Figure size
    """
    from sklearn.model_selection import learning_curve
    
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)
    
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, train_sizes=train_sizes, n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=figsize)
    plt.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    
    plt.plot(train_sizes_abs, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_prediction_distribution(y_true, y_pred, model_name="Model", figsize=(10, 6)):
    """
    Plot distribution of predictions vs actual values.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        model_name (str): Name of the model
        figsize (tuple): Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Actual distribution
    ax1.hist(y_true, bins=2, alpha=0.7, color='blue', label='Actual')
    ax1.set_title(f'Actual Distribution - {model_name}')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Count')
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['No Show', 'Show'])
    
    # Predicted distribution
    ax2.hist(y_pred, bins=2, alpha=0.7, color='red', label='Predicted')
    ax2.set_title(f'Predicted Distribution - {model_name}')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Count')
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['No Show', 'Show'])
    
    plt.tight_layout()
    plt.show()
