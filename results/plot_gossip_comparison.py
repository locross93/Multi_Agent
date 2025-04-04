import os
import json
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import argparse

def plot_gossip_comparison(models_dict, save_path=None, figsize=(18, 12)):
    """
    Create a 3x2 subplot comparing gossip experiment results across different models.
    
    Args:
        models_dict: Dictionary mapping model names to their results directories
                     Example: {'Claude 3 Opus': 'results/opus', 'GPT-4': 'results/gpt4'}
        save_path: Path to save the combined plot (None to not save)
        figsize: Size of the figure (width, height)
    """
    if len(models_dict) > 6:
        print(f"Warning: Only the first 6 models will be plotted (you provided {len(models_dict)})")
        # Limit to 6 models
        models_dict = dict(list(models_dict.items())[:6])
    
    # Create the figure and subplots
    fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)
    axes = axes.flatten()  # Flatten to make indexing easier
    
    # Define styles for conditions (consistent across all subplots)
    styles = {
        'basic': {'color': 'blue', 'marker': 'o', 'linestyle': '-', 'label': 'Basic (No Gossip)'},
        'gossip': {'color': 'green', 'marker': 's', 'linestyle': '--', 'label': 'Gossip Only'},
        'gossip-with-ostracism': {'color': 'red', 'marker': '^', 'linestyle': '-.', 'label': 'Gossip with Ostracism'}
    }
    
    # Process each model's data
    for i, (model_name, results_dir) in enumerate(models_dict.items()):
        if i >= 6:  # Safety check
            break
            
        ax = axes[i]
        
        # Load results for all conditions for this model
        results_files = list(pathlib.Path(results_dir).glob('gossip_exp_*.json'))
        
        if not results_files:
            ax.text(0.5, 0.5, f"No data found for\n{model_name}\nin {results_dir}", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            continue
            
        # Reference components to check for consistency
        reference_components = None
            
        # Group results by condition
        results_by_condition = defaultdict(list)
        for file_path in results_files:
            with open(file_path, 'r') as f:
                result = json.load(f)
                
                # Check model architecture consistency
                if 'config' in result and 'components' in result['config']:
                    if reference_components is None:
                        reference_components = result['config']['components']
                    else:
                        assert result['config']['components'] == reference_components, \
                            f"File {file_path} has different components than the reference"
                
                condition = result['condition']
                results_by_condition[condition].append(result)
        
        # Calculate average contributions per round by condition
        avg_contributions = {}
        se_contributions = {}  # For error bars
        for condition, results_list in results_by_condition.items():
            contributions_by_round = {str(r): [] for r in range(1, 7)}
            
            # Collect all contributions for each round
            for result in results_list:
                for round_key, contributions in result.get('contributions', {}).items():
                    if round_key in contributions_by_round:
                        contributions_by_round[round_key].extend(list(contributions.values()))
            
            # Calculate average and standard error for each round
            avg_by_round = []
            se_by_round = []
            for round_num in range(1, 7):
                round_key = str(round_num)
                if contributions_by_round[round_key]:
                    avg_by_round.append(np.mean(contributions_by_round[round_key]))
                    se_by_round.append(np.std(contributions_by_round[round_key]) / 
                                      np.sqrt(len(contributions_by_round[round_key])))
                else:
                    avg_by_round.append(0)
                    se_by_round.append(0)
            
            avg_contributions[condition] = avg_by_round
            se_contributions[condition] = se_by_round
        
        # Plot each condition on this subplot
        rounds = list(range(1, 7))
        for condition, avg_values in avg_contributions.items():
            if condition in styles:
                style = styles[condition]
                ax.errorbar(
                    rounds, 
                    avg_values, 
                    yerr=se_contributions[condition],
                    capsize=4,
                    linewidth=2.5,  # Thicker lines
                    markersize=8,    # Larger markers
                    **{k: v for k, v in style.items() if k != 'label'}  # Remove label for now
                )
        
        # Add title and labels to this subplot with increased font sizes
        ax.set_title(model_name, fontsize=18, fontweight='bold')
        ax.set_xlabel('Round', fontsize=18)
        ax.set_ylabel('Average Contribution ($)', fontsize=18)
        ax.set_xticks(rounds)
        ax.set_ylim(0, 10)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Increase tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Print some statistics for this model
        print(f"\n--- {model_name} ---")
        print("Average Contributions by Condition and Round:")
        for condition, avgs in avg_contributions.items():
            print(f"{condition}:")
            for round_num, avg in enumerate(avgs, 1):
                print(f"  Round {round_num}: ${avg:.2f}")
            print(f"  Overall: ${sum(avgs)/len(avgs):.2f}")
    
    # Create a single legend for the entire figure - Much larger now
    handles, labels = [], []
    for condition, style in styles.items():
        line, = plt.plot([], [], linewidth=3, markersize=10, **style)
        handles.append(line)
        labels.append(style['label'])
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.07),
               fontsize=16, frameon=True, fancybox=True, shadow=True, framealpha=0.9,
               borderpad=1, handletextpad=0.8, handlelength=2.5)
    
    # Add overall title
    fig.suptitle('Gossip Experiment Results Comparison Across Models', fontsize=24, fontweight='bold', y=1.05)
    
    # Save the figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nCombined plot saved to: {save_path}")
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])  # Adjusted to make more room for legend
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot comparison of gossip experiment results across models')
    parser.add_argument('--output', type=str, default='results/gossip_comparison.png',
                        help='Path to save the output figure')
    
    args = parser.parse_args()

    models_dict = {
            "Social architecture (ToM + Personas)": "results/gossip/03-26-2025_16-12",
            "Social + Strategic Reflection": "results/gossip/03-25-2025_21-06",
            "Social + Emotional Reflection": "results/gossip/03-26-2025_14-52",
            "Social + Strategy and Emotion Reflection": "results/gossip/03-26-2025_14-57",
            "No Theory of Mind": "results/gossip/03-26-2025_15-01",
            "No ToM or Personas": "results/gossip/03-26-2025_15-08"
            }
    
    plot_gossip_comparison(models_dict, args.output)