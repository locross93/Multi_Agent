# plot_tpp_results.py
import statsmodels.api as sm
import json
import pathlib
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
import os

def load_experiments(results_dir: str) -> List[Dict]:
    """Load all experiment files from the specified directory."""
    dir_path = pathlib.Path(results_dir)
    experiments = []
    
    # Check if there are any files with _public suffix
    public_files = list(dir_path.glob('tpp_exp_*_public.json'))
    
    if public_files:
        # If we have public files, load those
        for exp_file in public_files:
            with open(exp_file) as f:
                experiments.append(json.load(f))
    else:
        # Otherwise load all experiment files
        for exp_file in dir_path.glob('tpp_exp_*.json'):
            with open(exp_file) as f:
                experiments.append(json.load(f))
    
    return experiments

def analyze_experiments(experiments: List[Dict], model_name: str):
    """Extract relevant data for plotting with model name."""
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
                'model': model_name,
                'signaller_punished': did_punish,  # Whether this Recipient punished when they were Signaller
                'chooser_amount': amount_sent,
                'tripled_amount': tripled_amount,
                'returned_amount': amount_returned,
                'return_percentage': (amount_returned / tripled_amount) * 100 if tripled_amount > 0 else 0
            })
        except (KeyError, ValueError, AttributeError) as e:
            print(f"Skipping experiment {exp.get('config', {}).get('experiment_id', 'unknown')}: {str(e)}")
            continue
    
    if not data:
        print(f"Warning: No valid experiments found to analyze for model {model_name}!")
        return pd.DataFrame()
        
    return pd.DataFrame(data)

def create_plots(combined_data: pd.DataFrame, save_dir: pathlib.Path):
    """Create and save all plots with multiple models."""
    # Set seaborn style and palette
    sns.set_theme(style="whitegrid")
    
    # Define custom colors - cherry red for punisher, lighter red for non-punisher
    punisher_color = '#de2d2d'  # Cherry red
    non_punisher_color = '#ea9898'  # Lighter red
    custom_palette = {True: punisher_color, False: non_punisher_color}
    
    # 1. Percentage of signallers that punished by model
    plt.figure(figsize=(10, 6))
    
    # Calculate punishment rates by model
    punishment_rates = combined_data.groupby('model')['signaller_punished'].mean() * 100
    
    # Create bar plot
    ax = sns.barplot(x=punishment_rates.index, y=punishment_rates.values, 
                    color=punisher_color,
                    capsize=0.1)
    
    # Customize aesthetics
    plt.ylabel('Percentage of Signallers Who Punished (%)', fontsize=14)
    plt.xlabel('Model Architecture', fontsize=18)
    plt.xticks(fontsize=14, fontweight='bold', rotation=30, ha='right')
    plt.yticks(fontsize=16)
    plt.title('Punishment Rate by Model', fontsize=18, pad=20)
    plt.ylim(0, 100)
    
    # Add percentage values on bars
    for i, pct in enumerate(punishment_rates):
        ax.text(i, pct + 1, f'{pct:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    sns.despine(top=True, right=True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'tpp_punishment_percentage_by_model4.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 2. Average money sent by Chooser based on whether recipient (as signaller) punished
    plt.figure(figsize=(8, 6))
    
    # Group data by model and punishment status
    ax = sns.barplot(data=combined_data, x='model', y='chooser_amount', 
                   hue='signaller_punished', errorbar='se', palette=custom_palette)
    ax.grid(False)
    # Despine top and right
    sns.despine(top=True, right=True)
    
    # Set remaining spines to black
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    
    # Customize aesthetics
    plt.ylabel('Amount Sent by Chooser ($)', fontsize=16)
    plt.xlabel('Model Architecture', fontsize=18)
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14)
    plt.title('Amount Sent by Chooser vs Recipient\'s Previous Punishment', fontsize=16)
    
    # Adjust legend
    legend = plt.legend(title='Recipient Punished?', fontsize=14, title_fontsize=14)
    legend.get_texts()[0].set_text('No')
    legend.get_texts()[1].set_text('Yes')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'tpp_chooser_amounts_by_model4.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 3. Return percentage based on whether recipient (as signaller) punished
    plt.figure(figsize=(8, 6))
    
    # Group data by model and punishment status
    ax = sns.barplot(data=combined_data, x='model', y='return_percentage',
                   hue='signaller_punished', errorbar='se', palette=custom_palette)
    ax.grid(False)
    # Despine top and right
    sns.despine(top=True, right=True)
    
    # Set remaining spines to black
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    
    # Customize aesthetics
    plt.ylabel('Percentage of Tripled Amount Returned (%)', fontsize=14)
    plt.xlabel('Model Architecture', fontsize=18)
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14)
    plt.title('Return Percentage vs Recipient\'s Previous Punishment', fontsize=16)
    
    # Adjust legend - moved to bottom left
    legend = plt.legend(title='Recipient Punished?', fontsize=14, title_fontsize=14, 
                       loc='lower left')
    legend.get_texts()[0].set_text('No')
    legend.get_texts()[1].set_text('Yes')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'tpp_return_percentages_by_model4.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 4. Combined Return percentage vs received amount scatter
    plt.figure(figsize=(12, 8))
    
    ax = sns.scatterplot(data=combined_data, x='tripled_amount', y='return_percentage',
                        hue='signaller_punished', style='model', s=100,
                        palette=custom_palette)
    
    # Customize aesthetics
    plt.xlabel('Amount Received (Tripled Amount) ($)', fontsize=14)
    plt.ylabel('Percentage Returned (%)', fontsize=14)
    plt.title('Return Percentage vs Received Amount', fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Update legend
    plt.legend(fontsize=12, title_fontsize=12)
    
    sns.despine(top=True, right=True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'tpp_return_vs_received_by_model4.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Print statistical tests
    print("\nStatistical Tests by Model:")
    
    for model in combined_data['model'].unique():
        model_data = combined_data[combined_data['model'] == model]
        
        print(f"\n--- {model} ---")
        
        # T-test for chooser amounts
        try:
            t_stat, p_val = stats.ttest_ind(
                model_data[model_data['signaller_punished']]['chooser_amount'],
                model_data[~model_data['signaller_punished']]['chooser_amount']
            )
            print(f"Chooser Amounts T-test: t={t_stat:.3f}, p={p_val:.3f}")
        except:
            print("Could not perform t-test for chooser amounts (insufficient data)")

        # Linear regression for chooser amounts
        try:
            # Create dummy variable for 'signaller_punished'
            X = sm.add_constant(model_data['signaller_punished'].astype(int))
            
            # Convert dollar amounts to percentage of endowment ($10 endowment)
            # Multiply by 100/10 to convert from dollars to percentage of endowment
            y = model_data['chooser_amount'] * (100/10)
            
            # Run OLS regression with clustered standard errors
            model_ols = sm.OLS(y, X)
            
            # Use cluster-robust standard errors if multiple experiments
            if len(model_data['exp_id'].unique()) > 1:
                results = model_ols.fit(cov_type='cluster', 
                                    cov_kwds={'groups': model_data['exp_id']})
            else:
                results = model_ols.fit(use_t=True)
                
            coeff = results.params[1]  # Coefficient for signaller_punished
            std_err = results.bse[1]   # Standard error
            p_val = results.pvalues[1] # p-value
                
            print(f"Chooser Amounts Regression: coeff={coeff:.3f}, SE={std_err:.3f}, p={p_val:.3f}")
            print(f"  (Coefficient represents percentage points of endowment)")
        except Exception as e:
            print(f"Could not perform regression for chooser amounts: {str(e)}")
        
        # T-test for return percentages
        try:
            t_stat, p_val = stats.ttest_ind(
                model_data[model_data['signaller_punished']]['return_percentage'],
                model_data[~model_data['signaller_punished']]['return_percentage']
            )
            print(f"Return Percentages T-test: t={t_stat:.3f}, p={p_val:.3f}")
        except:
            print("Could not perform t-test for return percentages (insufficient data)")

        # Linear regression for return percentages
        try:
            # Create dummy variable for 'signaller_punished'
            X = sm.add_constant(model_data['signaller_punished'].astype(int))
            y = model_data['return_percentage']
            
            # Run OLS regression with clustered standard errors
            model_ols = sm.OLS(y, X)
            
            # Use cluster-robust standard errors if multiple experiments
            if len(model_data['exp_id'].unique()) > 1:
                results = model_ols.fit(cov_type='cluster', 
                                      cov_kwds={'groups': model_data['exp_id']})
            else:
                results = model_ols.fit(use_t=True)
                
            coeff = results.params[1]  # Coefficient for signaller_punished
            std_err = results.bse[1]   # Standard error
            p_val = results.pvalues[1] # p-value
                
            print(f"Return Percentages Regression: coeff={coeff:.3f}, SE={std_err:.3f}, p={p_val:.3f}")
        except Exception as e:
            print(f"Could not perform regression for return percentages: {str(e)}")

def main():
    # Dictionary of model names and their result directories
    models_dict = {
        "Theory of Mind + Personas\n (Social)": "results/tpp/12-31-2024_15-05",
        "Personas Only": "results/tpp/01-04-2025_18-39",
        "Strategy Reflection": "results/tpp/03-28-2025_17-20",
        "Emotion Reflection": "results/tpp/03-28-2025_14-14",
        # Add more models as needed
    }
    
    # Create combined data from all models
    all_data = []
    
    for model_name, results_dir in models_dict.items():
        print(f"Processing {model_name} from {results_dir}...")
        experiments = load_experiments(results_dir)
        model_data = analyze_experiments(experiments, model_name)
        all_data.append(model_data)
    
    # Combine all dataframes
    combined_data = pd.concat(all_data, ignore_index=True)
    
    if combined_data.empty:
        print("No valid data found for any model. Exiting.")
        return
    
    # Create plots directory in results
    plots_dir = pathlib.Path("results")
    plots_dir.mkdir(exist_ok=True)
    
    # Generate plots
    create_plots(combined_data, plots_dir)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    for model in combined_data['model'].unique():
        model_data = combined_data[combined_data['model'] == model]
        punish_rate = model_data['signaller_punished'].mean() * 100
        
        print(f"\n{model}:")
        print(f"Total experiments: {len(model_data)}")
        print(f"Punishment rate: {punish_rate:.1f}%")
        
        # Chooser amounts by punishment
        punish_chooser = model_data[model_data['signaller_punished']]['chooser_amount'].mean()
        no_punish_chooser = model_data[~model_data['signaller_punished']]['chooser_amount'].mean()
        print(f"Avg. chooser amount (punished): ${punish_chooser:.2f}")
        print(f"Avg. chooser amount (not punished): ${no_punish_chooser:.2f}")
        
        # Return percentages by punishment
        punish_return = model_data[model_data['signaller_punished']]['return_percentage'].mean()
        no_punish_return = model_data[~model_data['signaller_punished']]['return_percentage'].mean()
        print(f"Avg. return percentage (punished): {punish_return:.1f}%")
        print(f"Avg. return percentage (not punished): {no_punish_return:.1f}%")

        print("\nDetailed Statistics for Table:")
        
        # Calculate sent amounts in percentages (assuming $10 endowment)
        sent_punisher_pct = model_data[model_data['signaller_punished']]['chooser_amount'] * 10  # * 100/10
        sent_nonpunisher_pct = model_data[~model_data['signaller_punished']]['chooser_amount'] * 10  # * 100/10
        
        # Calculate return percentages (already in percentage scale)
        return_punisher = model_data[model_data['signaller_punished']]['return_percentage']
        return_nonpunisher = model_data[~model_data['signaller_punished']]['return_percentage']
        
        # Print table-ready statistics
        print(f"\n{model} Table Statistics:")
        print(f"Sent to punishers: M = {sent_punisher_pct.mean():.2f}, SD = {sent_punisher_pct.std():.2f}")
        print(f"Sent to non-punishers: M = {sent_nonpunisher_pct.mean():.2f}, SD = {sent_nonpunisher_pct.std():.2f}")
        print(f"Returned by punishers: M = {return_punisher.mean():.2f}, SD = {return_punisher.std():.2f}")
        print(f"Returned by non-punishers: M = {return_nonpunisher.mean():.2f}, SD = {return_nonpunisher.std():.2f}")

if __name__ == "__main__":
    main()