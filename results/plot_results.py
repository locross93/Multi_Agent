import json
import pathlib
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
def load_experiments(results_dir: str) -> List[Dict]:
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
            
            # Find Recipient's decision in round 2 (who was the Signaller in round 1)
            recipient_action = next((e for e in events if e['round'] == 2 and e['player'] == 'Recipient'), None)
            if recipient_action is None:
                continue
            amount_returned = float(recipient_action['action'].split('$')[1])
            tripled_amount = amount_sent * 3
            
            data.append({
                'exp_id': exp_id,
                'signaller_punished': did_punish,  # Whether this Recipient punished when they were Signaller
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
    """Create and save all plots."""
    # Set seaborn style and palette
    sns.set_theme(style="whitegrid")
    colors = sns.color_palette("Set2")
    
    # 1. Percentage of signallers that punished
    plt.figure(figsize=(8, 6))
    punishment_pct = (data['signaller_punished'].mean() * 100)
    sns.barplot(x=['Punished'], y=[punishment_pct], color=colors[0])
    plt.ylabel('Percentage of Signallers (%)')
    plt.title('Percentage of Signallers Who Punished in Round 1')
    plt.ylim(0, 100)
    plt.savefig(save_dir / 'punishment_percentage.png', bbox_inches='tight')
    plt.close()
    
    # 2. Average money sent by Chooser based on whether recipient (as signaller) punished
    plt.figure(figsize=(8, 6))
    sns.barplot(data=data, x='signaller_punished', y='chooser_amount', 
                errorbar='se', capsize=0.1, palette=colors[:2])
    plt.xticks([0, 1], ['Did Not Punish\nas Signaller', 'Punished\nas Signaller'])
    plt.xlabel('Recipient\'s Previous Punishment Decision')
    plt.ylabel('Amount Sent by Chooser ($)')
    plt.title('Amount Sent by Chooser vs Recipient\'s Previous Punishment')
    plt.savefig(save_dir / 'chooser_amounts.png', bbox_inches='tight')
    plt.close()
    
    # 3. Return percentage based on whether recipient (as signaller) punished
    plt.figure(figsize=(8, 6))
    sns.barplot(data=data, x='signaller_punished', y='return_percentage',
                errorbar='se', capsize=0.1, palette=colors[:2])
    plt.xticks([0, 1], ['Did Not Punish\nas Signaller', 'Punished\nas Signaller'])
    plt.xlabel('Recipient\'s Previous Punishment Decision')
    plt.ylabel('Percentage of Tripled Amount Returned (%)')
    plt.title('Return Percentage vs Recipient\'s Previous Punishment')
    plt.savefig(save_dir / 'return_percentages.png', bbox_inches='tight')
    plt.close()
    
    # 4. Return percentage vs received amount scatter
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='tripled_amount', y='return_percentage',
                    hue='signaller_punished', style='signaller_punished',
                    palette=colors[:2])
    plt.xlabel('Amount Received (Tripled Amount) ($)')
    plt.ylabel('Percentage Returned (%)')
    plt.title('Return Percentage vs Received Amount')
    # Update legend labels
    handles, _ = plt.gca().get_legend_handles_labels()
    plt.legend(handles, ['Did Not Punish as Signaller', 'Punished as Signaller'])
    plt.savefig(save_dir / 'return_vs_received.png', bbox_inches='tight')
    plt.close()

    # Print statistical tests
    print("\nStatistical Tests:")
    
    # T-test for chooser amounts
    t_stat, p_val = stats.ttest_ind(
        data[data['signaller_punished']]['chooser_amount'],
        data[~data['signaller_punished']]['chooser_amount']
    )
    print(f"\nChooser Amounts T-test:")
    print(f"t-statistic: {t_stat:.3f}")
    print(f"p-value: {p_val:.3f}")
    
    # T-test for return percentages
    t_stat, p_val = stats.ttest_ind(
        data[data['signaller_punished']]['return_percentage'],
        data[~data['signaller_punished']]['return_percentage']
    )
    print(f"\nReturn Percentages T-test:")
    print(f"t-statistic: {t_stat:.3f}")
    print(f"p-value: {p_val:.3f}")

def main():
    # Load experiments from the specified directory
    results_dir = "results/12-28-2024_13-04"
    experiments = load_experiments(results_dir)
    
    # Analyze data
    data = analyze_experiments(experiments)
    
    # Create plots directory
    plots_dir = pathlib.Path(results_dir) / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Generate plots
    create_plots(data, plots_dir)
    
    # Print summary statistics
    # print("\nSummary Statistics:")
    # print(f"Total experiments: {len(experiments)}")
    # print(f"Punishment rate: {data['punished'].mean():.1%}")
    # print("\nChooser amounts:")
    # print(data.groupby('punished')['chooser_amount'].describe())
    # print("\nReturn percentages:")
    # print(data.groupby('punished')['return_percentage'].describe())

if __name__ == "__main__":
    main()