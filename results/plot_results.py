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
    
    # Check if there are any files with _public/_private suffix
    #public_private_files = list(dir_path.glob('tpp_exp_*_public.json')) + list(dir_path.glob('tpp_exp_*_private.json'))
    public_private_files = list(dir_path.glob('tpp_exp_*_public.json'))
    
    if public_private_files:
        # If we have public/private files, load those
        for exp_file in public_private_files:
            with open(exp_file) as f:
                experiments.append(json.load(f))
    else:
        # Otherwise load all experiment files
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
    sns.set_theme(style="white")
    colors = sns.color_palette("Set2")
    
    # 1. Percentage of signallers that punished
    plt.figure(figsize=(4, 6))
    
    # Calculate statistics
    punishment_pct = (data['signaller_punished'].mean() * 100)
    n = len(data)
    se = np.sqrt((punishment_pct/100 * (1 - punishment_pct/100)) / n)
    ci = 1.96 * se * 100
    
    # Create bar plot
    ax = sns.barplot(x=['Overall'], y=[punishment_pct], 
                    color=colors[4],
                    width=1.5,
                    capsize=0.1)
    
    # Add error bars
    ax.errorbar(0, punishment_pct, yerr=ci,
                color='black', capsize=5, capthick=1)
    
    # Despine top and right
    sns.despine(top=True, right=True)
    ax.grid(False)
    
    # Set x-axis limits to make the bar appear thinner relative to plot
    plt.xlim(-2, 2)  # Extend x-axis on both sides
    
    # Customize aesthetics
    plt.ylabel('Percentage of Signallers Who Punished (%)', fontsize=14)
    plt.xlabel('')
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16)
    plt.title('Punishment Rate Comparison: Personas Needed to Not Punish', fontsize=24, pad=20)
    plt.ylim(0, 100)
    
    plt.savefig(save_dir / 'punishment_percentage2.png', bbox_inches='tight', dpi=300)
    plt.close()

    # 1.5 Comparison of punishment rates with/without personas
    plt.figure(figsize=(6, 6))
    
    # Create comparison data
    conditions = ['Base Agent', 'Full Model\n (w/ Personas)', 'Tom\nAblation']
    tom_ablation = 56.98924731182796
    percentages = [100, punishment_pct, tom_ablation]  # 100% for without personas
    print(punishment_pct)
    
    # Calculate CI for with personas (same as before)
    n = len(data)
    se = np.sqrt((punishment_pct/100 * (1 - punishment_pct/100)) / n)
    ci = 1.96 * se * 100
    ci_tom = 10.06235369824019
    
    # Create error bars array (no error bar for without personas condition)
    yerr = np.array([[0, ci]])  # [[lower errors], [upper errors]]
    
    # Create bar plot
    ax = sns.barplot(x=conditions, y=percentages,
                    palette=[colors[4], colors[4], colors[4]],  # Different colors for comparison
                    width=0.7,
                    capsize=0.1)
    
    # Add error bars manually (only for with personas condition)
    ax.errorbar(1, percentages[1], yerr=ci,
               color='black', capsize=5, capthick=1)
    ax.errorbar(2, percentages[2], yerr=ci_tom,
               color='black', capsize=5, capthick=1)
    
    # Despine and remove grid
    sns.despine(top=True, right=True)
    ax.grid(False)
    
    # Customize aesthetics
    plt.ylabel('Percentage of Signallers Who Punished (%)', fontsize=14)
    plt.xlabel('')
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16)
    plt.title('Punishment Rate Comparison', fontsize=24, pad=20)
    plt.ylim(0, 100)
    
    """ # Add exact percentages on top of bars
    for i, pct in enumerate(percentages):
        ax.text(i, pct + 2, f'{pct:.1f}%', 
                ha='center', va='bottom', fontsize=14, fontweight='bold') """
    
    plt.savefig(save_dir / 'punishment_percentage_comparison3.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 2. Average money sent by Chooser based on whether recipient (as signaller) punished
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(data=data, x='signaller_punished', y='chooser_amount', 
                    errorbar='se', capsize=0.1, palette=colors[:2])
    
    # Despine top and right
    sns.despine(top=True, right=True)
    ax.grid(False)
    
    plt.xticks([0, 1], ['Did Not Punish\nas Signaller', 'Punished\nas Signaller'], fontsize=16, fontweight='bold')
    plt.xlabel('Recipient\'s Previous Punishment Decision', fontsize=16)
    plt.ylabel('Amount Sent by Chooser ($)', fontsize=16)
    plt.title('Amount Sent by Chooser vs Recipient\'s Previous Punishment', fontsize=16)
    
    plt.savefig(save_dir / 'chooser_amounts2.png', bbox_inches='tight')
    plt.close()

    # print difference in chooser amounts between punished and not punished
    print(f"Difference in chooser amounts between punished and not punished: {data['chooser_amount'][data['signaller_punished']].mean() - data['chooser_amount'][~data['signaller_punished']].mean()}")
    
    # 3. Return percentage based on whether recipient (as signaller) punished
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(data=data, x='signaller_punished', y='return_percentage',
                    errorbar='se', capsize=0.1, palette=colors[:2])
    
    # Despine top and right
    sns.despine(top=True, right=True)
    ax.grid(False)
    
    plt.xticks([0, 1], ['Did Not Punish\nas Signaller', 'Punished\nas Signaller'], fontsize=16, fontweight='bold')
    plt.xlabel('Recipient\'s Previous Punishment Decision', fontsize=16)
    plt.ylabel('Percentage of Tripled Amount Returned (%)', fontsize=16)
    plt.title('Return Percentage vs Recipient\'s Previous Punishment', fontsize=16)
    
    plt.savefig(save_dir / 'return_percentages2.png', bbox_inches='tight')
    plt.close()
    
    # 4. Return percentage vs received amount scatter
    plt.figure(figsize=(10, 6))
    ax = sns.scatterplot(data=data, x='tripled_amount', y='return_percentage',
                        hue='signaller_punished', style='signaller_punished',
                        palette=colors[:2])
    
    # Despine top and right
    sns.despine(top=True, right=True)
    ax.grid(False)
    
    plt.xlabel('Amount Received (Tripled Amount) ($)')
    plt.ylabel('Percentage Returned (%)')
    plt.title('Return Percentage vs Received Amount')
    # Update legend labels
    handles, _ = plt.gca().get_legend_handles_labels()
    plt.legend(handles, ['Did Not Punish as Signaller', 'Punished as Signaller'])
    
    plt.savefig(save_dir / 'return_vs_received2.png', bbox_inches='tight')
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
    #results_dir = "results/12-28-2024_12-55"
    #results_dir = "results/12-28-2024_13-04"
    #results_dir = "results/12-28-2024_17-35"
    #results_dir = "results/12-31-2024_15-05" # main results
    # results_dir = "results/01-04-2025_18-39" $ tom ablation
    #results_dir = "results/tpp/03-28-2025_10-18"
    #results_dir = "results/tpp/03-28-2025_12-19"
    #results_dir = "results/tpp/03-28-2025_14-18"
    #results_dir = "results/tpp/03-28-2025_14-14" # emotion reflection
    results_dir = "results/tpp/03-28-2025_17-20" # strategy reflection
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