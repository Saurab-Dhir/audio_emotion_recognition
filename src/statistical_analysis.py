import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from file."""
    import yaml
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def load_cv_results(result_path):
    """Load cross-validation results."""
    if os.path.exists(result_path):
        with open(result_path, 'rb') as f:
            return pickle.load(f)
    else:
        logger.error(f"CV results file not found at {result_path}")
        return None

def perform_significance_test(cv_results, alpha=0.05):
    """
    Perform statistical significance tests between models.
    
    Args:
        cv_results (dict): Dictionary containing cross-validation results
        alpha (float): Significance level
        
    Returns:
        dict: Statistical test results
    """
    logger.info("Performing statistical significance tests...")
    
    models = list(cv_results.keys())
    metrics = ['accuracy', 'f1']
    
    # Create a table to store p-values
    results = {}
    
    for metric in metrics:
        results[metric] = {}
        
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i < j:  # Avoid redundant comparisons
                    # Get scores for both models
                    scores1 = np.array(cv_results[model1][metric])
                    scores2 = np.array(cv_results[model2][metric])
                    
                    # Paired t-test
                    t_stat, p_value = stats.ttest_rel(scores1, scores2)
                    
                    # Store results
                    model_pair = f"{model1}_vs_{model2}"
                    results[metric][model_pair] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < alpha,
                        'better_model': model1 if t_stat > 0 else model2 if t_stat < 0 else 'equal'
                    }
                    
                    # Log results
                    logger.info(f"t-test {model1} vs {model2} ({metric}): t={t_stat:.4f}, p={p_value:.4f}")
                    if p_value < alpha:
                        better = model1 if t_stat > 0 else model2
                        logger.info(f"  Significant difference: {better} performs better")
                    else:
                        logger.info("  No significant difference")
    
    return results

def bootstrap_confidence_intervals(cv_results, n_bootstrap=1000, alpha=0.05):
    """
    Calculate bootstrap confidence intervals for model performance.
    
    Args:
        cv_results (dict): Dictionary containing cross-validation results
        n_bootstrap (int): Number of bootstrap samples
        alpha (float): Significance level
        
    Returns:
        dict: Bootstrap confidence intervals
    """
    logger.info("Calculating bootstrap confidence intervals...")
    
    bootstrap_results = {}
    metrics = ['accuracy', 'f1']
    
    for model_name, model_results in cv_results.items():
        bootstrap_results[model_name] = {}
        
        for metric in metrics:
            scores = np.array(model_results[metric])
            bootstrap_samples = []
            
            # Generate bootstrap samples
            for _ in range(n_bootstrap):
                # Sample with replacement
                sample = np.random.choice(scores, size=len(scores), replace=True)
                bootstrap_samples.append(np.mean(sample))
            
            # Calculate confidence intervals
            lower = np.percentile(bootstrap_samples, alpha/2 * 100)
            upper = np.percentile(bootstrap_samples, (1 - alpha/2) * 100)
            
            bootstrap_results[model_name][metric] = {
                'mean': np.mean(scores),
                'lower_ci': lower,
                'upper_ci': upper
            }
            
            logger.info(f"{model_name} {metric}: {np.mean(scores):.4f} [{lower:.4f}, {upper:.4f}]")
    
    return bootstrap_results

def plot_model_comparison_with_ci(bootstrap_results, output_path=None):
    """
    Plot model comparison with confidence intervals.
    
    Args:
        bootstrap_results (dict): Bootstrap confidence intervals
        output_path (str): Path to save the plot
    """
    logger.info("Creating model comparison plot with confidence intervals...")
    
    metrics = ['accuracy', 'f1']
    models = list(bootstrap_results.keys())
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 6))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        means = []
        errors = []
        names = []
        
        for model in models:
            result = bootstrap_results[model][metric]
            means.append(result['mean'])
            errors.append([result['mean'] - result['lower_ci'], result['upper_ci'] - result['mean']])
            names.append(model.replace('_', ' ').title())
        
        errors = np.array(errors).T
        
        # Create bar plot with error bars
        ax.bar(range(len(means)), means, yerr=errors, capsize=10, 
               color=['#2C7BB6', '#D7191C'], alpha=0.7)
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_title(f"{metric.title()} with 95% CI")
        ax.set_ylim(min(means) * 0.95, max(means) * 1.05)
        
        # Add value labels on top of bars
        for j, v in enumerate(means):
            ax.text(j, v + errors[0][j] + 0.01, f"{v:.3f}", ha='center')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved model comparison plot to {output_path}")
    
    plt.close()

def create_significance_table(sig_results, output_path=None):
    """
    Create a table showing significance test results.
    
    Args:
        sig_results (dict): Statistical test results
        output_path (str): Path to save the table
    """
    logger.info("Creating significance test results table...")
    
    metrics = list(sig_results.keys())
    
    for metric in metrics:
        model_pairs = list(sig_results[metric].keys())
        data = []
        
        for pair in model_pairs:
            result = sig_results[metric][pair]
            data.append({
                'Model Comparison': pair,
                't-statistic': f"{result['t_statistic']:.4f}",
                'p-value': f"{result['p_value']:.4f}",
                'Significant': 'Yes' if result['significant'] else 'No',
                'Better Model': result['better_model'] if result['significant'] else '-'
            })
        
        df = pd.DataFrame(data)
        
        if output_path:
            table_path = output_path.replace('.csv', f'_{metric}.csv')
            df.to_csv(table_path, index=False)
            logger.info(f"Saved {metric} significance table to {table_path}")
        
        print(f"\n{metric.upper()} SIGNIFICANCE TEST RESULTS:")
        print(df)

def run_statistical_analysis(cv_results_path=None, output_dir=None):
    """
    Run statistical analysis on cross-validation results.
    
    Args:
        cv_results_path (str, optional): Path to cross-validation results
        output_dir (str, optional): Directory to save results
    """
    # Load configuration
    config = load_config()
    
    # Set default paths if not provided
    if cv_results_path is None:
        cv_results_path = os.path.join(config['paths']['results'], 'cv_results.pkl')
    
    if output_dir is None:
        output_dir = os.path.join(config['paths']['results'], 'statistical_analysis')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load cross-validation results
    cv_results = load_cv_results(cv_results_path)
    if cv_results is None:
        logger.error("Failed to load cross-validation results. Exiting.")
        return
    
    # Perform statistical significance testing
    sig_results = perform_significance_test(cv_results)
    
    # Calculate bootstrap confidence intervals
    bootstrap_results = bootstrap_confidence_intervals(cv_results)
    
    # Create visualization of model comparison with confidence intervals
    plot_path = os.path.join(output_dir, 'model_comparison_with_ci.png')
    plot_model_comparison_with_ci(bootstrap_results, plot_path)
    
    # Create significance test results table
    table_path = os.path.join(output_dir, 'significance_test_results.csv')
    create_significance_table(sig_results, table_path)
    
    logger.info("Statistical analysis completed successfully!")
    return {
        'significance_results': sig_results,
        'bootstrap_results': bootstrap_results
    }

if __name__ == "__main__":
    run_statistical_analysis() 