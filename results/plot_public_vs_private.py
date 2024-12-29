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
    sns.set_theme(style="whitegrid")
    colors = sns.color_palette("Set2")
    
    # 1. Percentage of signallers that punished by condition
    plt.figure(figsize=(8, 6))
    punishment_by_condition = data.groupby('condition')['signaller_punished'].mean() * 100
    sns.barplot(x=punishment_by_condition.index, y=punishment_by_condition.values, 
                palette=colors[:2])
    plt.ylabel('Percentage of Signallers Who Punished (%)')
    plt.xlabel('Condition')
    plt.title('Punishment Rate by Condition')
    plt.ylim(0, 100)
    plt.savefig(save_dir / 'punishment_by_condition.png', bbox_inches='tight')
    plt.close()
    
    # 2. Amount sent by Chooser by condition and punishment
    plt.figure(figsize=(10, 6))
    sns.barplot(data=data, x='condition', y='chooser_amount', 
                hue='signaller_punished', palette=colors[:2],
                errorbar='se', capsize=0.1)
    plt.xlabel('Condition')
    plt.ylabel('Amount Sent by Chooser ($)')
    plt.title('Amount Sent by Chooser vs Condition and Punishment')
    plt.legend(title='Recipient Punished', labels=['No', 'Yes'])
    plt.savefig(save_dir / 'chooser_amounts_by_condition.png', bbox_inches='tight')
    plt.close()
    
    # 3. Return percentage by condition and punishment
    plt.figure(figsize=(10, 6))
    sns.barplot(data=data, x='condition', y='return_percentage',
                hue='signaller_punished', palette=colors[:2],
                errorbar='se', capsize=0.1)
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

    # Statistical tests
    print("\nStatistical Tests:")
    
    # Chi-square test for punishment rates between conditions
    contingency = pd.crosstab(data['condition'], data['signaller_punished'])
    chi2, p_val = stats.chi2_contingency(contingency)[:2]
    print(f"\nPunishment Rate Chi-square test:")
    print(f"chi2-statistic: {chi2:.3f}")
    print(f"p-value: {p_val:.3f}")
    
    # Two-way ANOVA for chooser amounts
    from statsmodels.stats.anova import anova_lm
    from statsmodels.formula.api import ols
    
    model = ols('chooser_amount ~ C(condition) + C(signaller_punished) + C(condition):C(signaller_punished)', 
                data=data).fit()
    aov_table = anova_lm(model, typ=2)
    print("\nTwo-way ANOVA for Chooser Amounts:")
    print(aov_table)

def main():
    results_dir = "results/12-28-2024_17-35"
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