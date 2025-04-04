# plot_gossip_results.py
import os
import json
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import argparse

def plot_gossip_results(results_dir='results', save=True):
    """
    Analyze gossip experiment data and create line plots for contributions by round
    across different conditions.
    
    Args:
        results_dir: Directory where result JSON files are stored
    """
    # Load results for all conditions
    results_files = list(pathlib.Path(results_dir).glob('gossip_exp_*.json'))
    
    if not results_files:
        print(f"No result files found in {results_dir}")
        return
    
    print(f"Found {len(results_files)} result files")
    
    # Group results by condition
    results_by_condition = defaultdict(list)
    for file_path in results_files:
        with open(file_path, 'r') as f:
            result = json.load(f)
            if 'discussion' in str(file_path):
                condition = result['condition'] + '-and-discussion'
            else:
                condition = result['condition']
            results_by_condition[condition].append(result)
    
    # Calculate average contributions per round by condition
    avg_contributions = {}
    se_contributions = {}  # For error bars (changed from std to se)
    for condition, results_list in results_by_condition.items():
        contributions_by_round = {str(r): [] for r in range(1, 7)}
        
        # Collect all contributions for each round
        for result in results_list:
            for round_key, contributions in result.get('contributions', {}).items():
                if round_key in contributions_by_round:
                    contributions_by_round[round_key].extend(list(contributions.values()))
        
        # Calculate average and standard error for each round
        avg_by_round = []
        se_by_round = []  # Changed from std to se
        for round_num in range(1, 7):
            round_key = str(round_num)
            if contributions_by_round[round_key]:
                avg_by_round.append(np.mean(contributions_by_round[round_key]))
                # Calculate standard error instead of standard deviation
                se_by_round.append(np.std(contributions_by_round[round_key]) / 
                                  np.sqrt(len(contributions_by_round[round_key])))
            else:
                avg_by_round.append(0)
                se_by_round.append(0)
        
        avg_contributions[condition] = avg_by_round
        se_contributions[condition] = se_by_round  # Changed from std to se
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Define colors and markers for different conditions
    styles = {
        'basic': {'color': 'blue', 'marker': 'o', 'linestyle': '-', 'label': 'Basic (No Gossip)'},
        'gossip': {'color': 'green', 'marker': 's', 'linestyle': '--', 'label': 'Gossip Only'},
        'gossip-with-ostracism': {'color': 'red', 'marker': '^', 'linestyle': '-.', 'label': 'Gossip with Ostracism'},
        'gossip-with-ostracism-and-discussion': {'color': 'purple', 'marker': '^', 'linestyle': '--', 'label': 'Gossip with Ostracism and Discussion'}
    }
    
    # Plot each condition
    rounds = list(range(1, 7))
    for condition, avg_values in avg_contributions.items():
        if condition in styles:
            style = styles[condition]
            plt.errorbar(
                rounds, 
                avg_values, 
                yerr=se_contributions[condition],  # Changed from std to se
                capsize=4,
                **style
            )
    
    # Add labels and title
    plt.xlabel('Round')
    plt.ylabel('Average Contribution ($)')
    plt.title('Average Contributions by Round and Condition')
    plt.xticks(rounds)
    plt.ylim(0, 10)  # Assuming max contribution is $10
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Print some statistics
    print("\nAverage Contributions by Condition and Round:")
    for condition, avgs in avg_contributions.items():
        print(f"{condition}:")
        for round_num, avg in enumerate(avgs, 1):
            print(f"  Round {round_num}: ${avg:.2f}")
        print(f"  Overall: ${sum(avgs)/len(avgs):.2f}")
    
    # Save the plot
    if save:
        output_path = os.path.join(results_dir, 'gossip_contributions_by_round.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
    
    # Show the plot
    plt.show()
    
    # Additional analysis: Calculate and print condition effects
    if 'basic' in avg_contributions and len(avg_contributions) > 1:
        print("\nCondition Effects (Relative to Basic Condition):")
        basic_overall = sum(avg_contributions['basic']) / len(avg_contributions['basic'])
        
        for condition, avgs in avg_contributions.items():
            if condition != 'basic':
                condition_overall = sum(avgs) / len(avgs)
                diff = condition_overall - basic_overall
                percent = (diff / basic_overall) * 100
                print(f"  {condition}: {diff:.2f} (${condition_overall:.2f} vs ${basic_overall:.2f}, {percent:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot gossip experiment results')
    parser.add_argument('--results_dir', type=str, default='results', 
                        help='Directory containing experiment result files')
    args = parser.parse_args()
    
    plot_gossip_results(args.results_dir) 