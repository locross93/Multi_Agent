import json
import pathlib
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

def load_all_experiments(results_dir: str) -> List[Dict]:
    """Load all experiment files from the specified directory."""   
    dir_path = pathlib.Path(results_dir)
    experiments = []
    
    for exp_file in dir_path.glob('tpp_exp_*.json'):
        with open(exp_file) as f:
            experiments.append(json.load(f))
    
    return experiments

def analyze_experiments(experiments: List[Dict]):
    """Extract relevant data for plotting."""
    data = []
    
    for exp in experiments:
        try:
            exp_id = exp['config']['experiment_id']
            events = exp['events']
            # Get condition from either config or events
            is_public = exp['config'].get('public_condition', 
                                        events[0].get('condition') == 'public')
            condition = 'Public' if is_public else 'Private'
            
            # Find punishment decision in round 1
            punishment = next((e for e in events if e['round'] == 1), None)
            if punishment is None:
                continue
            did_punish = punishment['reward'] == -2.0
            
            # Find Chooser's decision in round 2
            chooser_action = next((e for e in events if e['round'] == 2 and e['player'] == 'Chooser'), None)
            if chooser_action is None:
                continue
            amount_sent = float(chooser_action['action'].split('$')[1])
            
            # Find Recipient's decision in round 2
            recipient_action = next((e for e in events if e['round'] == 2 and e['player'] == 'Recipient'), None)
            if recipient_action is None:
                continue
            amount_returned = float(recipient_action['action'].split('$')[1])
            tripled_amount = amount_sent * 3
            
            data.append({
                'exp_id': exp_id,
                'condition': condition,
                'signaller_punished': did_punish,
                'chooser_amount': amount_sent,
                'tripled_amount': tripled_amount,
                'returned_amount': amount_returned,
                'return_percentage': (amount_returned / tripled_amount) * 100
            })
            
        except (KeyError, ValueError, AttributeError) as e:
            print(f"Skipping experiment {exp.get('config', {}).get('experiment_id', 'unknown')}: {str(e)}")
            continue
    
    if not data:
        raise ValueError("No valid experiments found to analyze!")
        
    return pd.DataFrame(data)

def create_plots(data: pd.DataFrame, save_dir: pathlib.Path):
    """Create and save all plots comparing public vs private conditions."""
    sns.set_theme(style="white")
    colors = sns.color_palette("Set2")
    
    # # 1. Percentage of signallers that punished by condition
    # plt.figure(figsize=(8, 6))
    # punishment_by_condition = data.groupby('condition')['signaller_punished'].mean() * 100
    # sns.barplot(x=punishment_by_condition.index, y=punishment_by_condition.values, 
    #             palette=colors[:2])
    # plt.ylabel('Percentage of Signallers Who Punished (%)')
    # plt.xlabel('Condition')
    # plt.title('Punishment Rate by Condition')
    # plt.ylim(0, 100)
    # plt.savefig(save_dir / 'punishment_by_condition.png', bbox_inches='tight')
    # plt.close()

    # 1. Percentage of signallers that punished by condition
    plt.figure(figsize=(8, 6))
    
    # Calculate statistics for each condition
    condition_stats = []  # Renamed from stats_by_condition
    for condition in ['Public', 'Private']:
        condition_data = data[data['condition'] == condition]['signaller_punished']
        n = len(condition_data)
        mean = condition_data.mean() * 100
        se = np.sqrt((mean/100 * (1 - mean/100)) / n)  # Standard error for proportion
        ci = 1.96 * se * 100  # 95% CI
        condition_stats.append({  # Using new name
            'condition': condition,
            'mean': mean,
            'ci': ci
        })
    
    # Create DataFrame for plotting
    plot_data = pd.DataFrame(condition_stats)  # Using new name
    
    # Create bar plot with error bars
    ax = sns.barplot(data=plot_data, x='condition', y='mean', 
                    palette=colors[:2],
                    capsize=0.1)
    
    # Add error bars manually
    for i, row in plot_data.iterrows():
        ax.errorbar(i, row['mean'], yerr=row['ci'], 
                   color='black', capsize=5, capthick=1)

    # Despine top and right
    sns.despine(top=True, right=True)
    ax.grid(False)
    
    plt.ylabel('Percentage of Signallers Who Punished (%)', fontsize=14)
    plt.xlabel('Condition', fontsize=18)
    # make labels fontsize 18 and bold
    plt.xticks(fontsize=18, fontweight='bold')
    plt.title('Punishment Rate by Condition', fontsize=20)
    plt.ylim(0, 100)
    
    # Run statistical tests
    contingency = pd.crosstab(data['condition'], data['signaller_punished'])
    
    # Chi-square test
    chi2, chi2_p = stats.chi2_contingency(contingency)[:2]
    
    # Fisher's exact test
    fisher_p = stats.fisher_exact(contingency)[1]
    
    # Add statistical test results to plot
    chi2_text = f'p < 0.001' if chi2_p < 0.001 else f'p = {chi2_p:.3f}'
    #fisher_text = f'p < 0.001' if fisher_p < 0.001 else f'p = {fisher_p:.3f}'
    
    plt.text(0.05, 0.95, 
             f'Chi-square test: {chi2_text}',
             #f'Fisher\'s exact test: {fisher_text}',
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.savefig(save_dir / 'punishment_by_condition.png', bbox_inches='tight')
    plt.close()
    
    # Print detailed statistics
    print("\nPunishment Rate Analysis:")
    for stat in condition_stats:  # Using new name
        condition = stat['condition']
        mean = stat['mean']
        ci = stat['ci']
        n = len(data[data['condition'] == condition])
        print(f"\n{condition} condition:")
        print(f"N = {n}")
        print(f"Punishment rate: {mean:.1f}%")
        print(f"95% CI: [{mean-ci:.1f}%, {mean+ci:.1f}%]")
    
    print(f"\nChi-square test:")
    print(f"χ² = {chi2:.3f}")
    print(f"p = {chi2_p:.3f}")
    
    print(f"\nFisher's exact test:")
    print(f"p = {fisher_p:.3f}")
    
    # 2. Amount sent by Chooser by condition and punishment
    plt.figure(figsize=(10, 6))
    sns.barplot(data=data, x='condition', y='chooser_amount', 
                hue='signaller_punished', palette=colors[:2],
                errorbar='se', capsize=0.1)
    
    # Despine top and right
    sns.despine(top=True, right=True)
    ax.grid(False)
    
    plt.xlabel('Condition', fontsize=18)
    plt.ylabel('Amount Sent by Chooser ($)', fontsize=18)
    plt.title('Amount Sent by Chooser vs Condition and Punishment', fontsize=20)
    plt.xticks(fontsize=18, fontweight='bold')
    plt.legend(title='Recipient Punished')
    #plt.legend(title='Recipient Punished', labels=['No', 'Yes'])
    plt.savefig(save_dir / 'chooser_amounts_by_condition.png', bbox_inches='tight')
    plt.close()
    
    # 3. Return percentage by condition and punishment
    plt.figure(figsize=(10, 6))
    sns.barplot(data=data, x='condition', y='return_percentage',
                hue='signaller_punished', palette=colors[:2],
                errorbar='se', capsize=0.1)
    
    # Despine top and right
    sns.despine(top=True, right=True)
    ax.grid(False)
    
    plt.xlabel('Condition')
    plt.ylabel('Percentage of Tripled Amount Returned (%)')
    plt.title('Return Percentage vs Condition and Punishment')
    plt.legend(title='Recipient Punished', labels=['No', 'Yes'])
    plt.savefig(save_dir / 'return_percentages_by_condition.png', bbox_inches='tight')
    plt.close()
    
    # 4. Return percentage vs received amount scatter by condition
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=data, x='tripled_amount', y='return_percentage',
                    hue='condition', style='signaller_punished',
                    palette=colors[:2])
    plt.xlabel('Amount Received (Tripled Amount) ($)')
    plt.ylabel('Percentage Returned (%)')
    plt.title('Return Percentage vs Received Amount by Condition')
    plt.legend(title='Condition & Punishment',
              labels=['Public & No Punishment', 'Public & Punished',
                     'Private & No Punishment', 'Private & Punished'])
    plt.savefig(save_dir / 'return_vs_received_by_condition.png', bbox_inches='tight')
    plt.close()

    # # Statistical tests
    # print("\nStatistical Tests:")
    
    # # Chi-square test for punishment rates between conditions
    # contingency = pd.crosstab(data['condition'], data['signaller_punished'])
    # chi2, p_val = stats.chi2_contingency(contingency)[:2]
    # print(f"\nPunishment Rate Chi-square test:")
    # print(f"chi2-statistic: {chi2:.3f}")
    # print(f"p-value: {p_val:.3f}")
    
    # # Two-way ANOVA for chooser amounts
    # from statsmodels.stats.anova import anova_lm
    # from statsmodels.formula.api import ols
    
    # model = ols('chooser_amount ~ C(condition) + C(signaller_punished) + C(condition):C(signaller_punished)', 
    #             data=data).fit()
    # aov_table = anova_lm(model, typ=2)
    # print("\nTwo-way ANOVA for Chooser Amounts:")
    # print(aov_table)

def main():
    #results_dir = "results/12-28-2024_17-35"
    results_dir = "results/12-31-2024_15-05"
    experiments = load_all_experiments(results_dir)
    
    data = analyze_experiments(experiments)
    
    plots_dir = pathlib.Path(results_dir) / 'plots_comparison'
    plots_dir.mkdir(exist_ok=True)
    
    create_plots(data, plots_dir)
    
    # Print summary statistics
    print("\nSummary Statistics by Condition:")
    print(data.groupby('condition').agg({
        'signaller_punished': ['count', 'mean'],
        'chooser_amount': ['mean', 'std'],
        'return_percentage': ['mean', 'std']
    }).round(3))

if __name__ == "__main__":
    main() 