# statistical_analysis.py
import json
import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def load_data(results_dir):
    """Load the three JSON files with experimental data"""
    conditions = ['basic', 'gossip', 'gossip-with-ostracism']
    data = {}
    
    for condition in conditions:
        file_path = os.path.join(results_dir, f'gossip_exp_0_{condition}.json')
        try:
            with open(file_path, 'r') as f:
                data[condition] = json.load(f)
            print(f"Successfully loaded {file_path}")
            # Debug: Print the top-level keys to understand the structure
            print(f"Keys in {condition} data: {list(data[condition].keys())}")
            if 'contributions' in data[condition]:
                # Check the rounds
                rounds = list(data[condition]['contributions'].keys())
                print(f"Rounds in {condition}: {rounds[:3]}{'...' if len(rounds) > 3 else ''}")
                # Check the first round to understand player format
                if rounds:
                    first_round = rounds[0]
                    players = list(data[condition]['contributions'][first_round].keys())
                    print(f"Players in round {first_round}: {players[:3]}{'...' if len(players) > 3 else ''}")
        except FileNotFoundError:
            print(f"Warning: Could not find {file_path}")
    
    return data

def extract_contributions(data):
    """
    Extract contributions by player, round, and condition
    Returns a DataFrame with columns: player, round, condition, contribution
    """
    records = []
    
    for condition, condition_data in data.items():
        contributions = condition_data.get('contributions', {})
        
        for round_num, round_data in contributions.items():
            for player, contribution in round_data.items():
                records.append({
                    'player': player,
                    'round': int(round_num),
                    'condition': condition,
                    'contribution': float(contribution)
                })
    
    return pd.DataFrame(records)

def calculate_total_contributions(df):
    """Calculate the total contribution for each player in each condition"""
    return df.groupby(['player', 'condition'])['contribution'].sum().reset_index()

def test_between_conditions(df):
    """
    Perform within-subjects F-test to compare total contributions between conditions
    """
    # Calculate the total contribution for each player in each condition
    player_totals = calculate_total_contributions(df)
    
    # Get unique players and conditions
    players = df['player'].unique()
    conditions = df['condition'].unique()
    
    # Print summary statistics
    print("\nSummary Statistics for Total Contributions by Condition:")
    condition_stats = {}
    for condition in conditions:
        condition_data = player_totals[player_totals['condition'] == condition]
        mean = condition_data['contribution'].mean()
        std = condition_data['contribution'].std()
        condition_stats[condition] = {'mean': mean, 'std': std}
        print(f"{condition}: Mean = {mean:.2f}, SD = {std:.2f}")
    
    # 1. Test for overall differences between conditions (F-test)
    # Reshape data to have one row per player and columns for each condition
    wide_data = player_totals.pivot(index='player', columns='condition', values='contribution')
    
    # Drop any players with missing data
    wide_data = wide_data.dropna()
    
    # Perform F-test (ANOVA)
    f_stat, p_value = stats.f_oneway(*[wide_data[condition] for condition in conditions])
    
    print("\nANOVA Results (Between Conditions):")
    print(f"F-statistic: {f_stat:.2f}, p-value: {p_value:.6f}")
    
    # 2. Pairwise comparisons
    print("\nPairwise Comparisons (t-tests):")
    for i, cond1 in enumerate(conditions):
        for j, cond2 in enumerate(conditions):
            if j > i:  # Avoid duplicates
                t_stat, p_val = stats.ttest_rel(wide_data[cond1], wide_data[cond2])
                print(f"{cond1} vs {cond2}: t = {t_stat:.2f}, p = {p_val:.6f}")
    
    return f_stat, p_value, condition_stats

def test_linear_trends(df):
    """
    Test for linear trends in contributions across rounds within each condition
    """
    # Get the mean contribution for each round in each condition
    round_means = df.groupby(['condition', 'round'])['contribution'].mean().reset_index()
    
    # Analyze the linear trend for each condition
    conditions = df['condition'].unique()
    
    print("\nLinear Trend Analysis by Condition:")
    
    results = {}
    
    for condition in conditions:
        condition_data = round_means[round_means['condition'] == condition]
        
        # Prepare X (round numbers) and Y (mean contributions)
        x = condition_data['round'].values
        y = condition_data['contribution'].values
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Calculate F-statistic for the linear trend
        # The F-statistic is the square of the t-statistic for the slope
        dof = len(x) - 2  # Degrees of freedom
        f_stat = (slope / std_err)**2
        f_p_value = 1 - stats.f.cdf(f_stat, 1, dof)
        
        # Save results
        results[condition] = {
            'slope': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'f_stat': f_stat,
            'f_p_value': f_p_value,
            'intercept': intercept,
            'std_err': std_err
        }
        
        print(f"\nCondition: {condition}")
        print(f"Slope: {slope:.4f} (points per round)")
        print(f"F-statistic for linear trend: {f_stat:.2f}")
        print(f"p-value: {p_value:.6f}")
        
        if p_value < 0.05:
            direction = "increasing" if slope > 0 else "decreasing"
            print(f"There is a significant {direction} linear trend.")
        else:
            print("There is no significant linear trend.")
    
    # Compare linear trends between conditions
    print("\nComparison of Linear Trends Between Conditions:")
    for i, cond1 in enumerate(conditions):
        for j, cond2 in enumerate(conditions):
            if j > i:  # Avoid duplicates
                # Test for differences in slopes
                t_stat = (results[cond1]['slope'] - results[cond2]['slope']) / np.sqrt(
                    results[cond1]['std_err']**2 + results[cond2]['std_err']**2)
                # Calculate p-value from t-statistic (with appropriate degrees of freedom)
                dof = 2 * (6 - 2)  # 2 * (n - 2)
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), dof))
                
                print(f"{cond1} vs {cond2} (slope comparison): t = {t_stat:.2f}, p = {p_val:.6f}")
    
    return results, round_means

def plot_contributions_by_round(round_means):
    """Plot the mean contributions by round for each condition"""
    conditions = round_means['condition'].unique()
    
    plt.figure(figsize=(10, 6))
    
    # Define nicer colors and markers
    colors = ['#E41A1C', '#377EB8', '#4DAF4A']
    markers = ['o', 's', '^']
    
    for i, condition in enumerate(conditions):
        condition_data = round_means[round_means['condition'] == condition]
        plt.plot(condition_data['round'], condition_data['contribution'], 
                 marker=markers[i], color=colors[i], linewidth=2, markersize=8, 
                 label=condition)
    
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Mean Contribution', fontsize=12)
    plt.title('Mean Contributions by Round and Condition', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Set the x-axis to show only integer values
    plt.xticks(range(1, 7))
    
    plt.savefig('contributions_by_condition.png', dpi=300)
    plt.close()

def main(results_dir):
    """Main function to run all analyses"""
    # Load the data
    data = load_data(results_dir)
    
    # Extract contributions
    df = extract_contributions(data)
    
    print(f"Total number of contributions records: {len(df)}")
    print(f"Number of unique players: {df['player'].nunique()}")
    
    # Test 1: F-tests for differences between conditions
    print("\n=== ANALYSIS 1: DIFFERENCES BETWEEN CONDITIONS ===")
    f_stat, p_value, condition_stats = test_between_conditions(df)
    
    # Test 2: Linear trend analyses within each condition
    print("\n=== ANALYSIS 2: LINEAR TRENDS WITHIN CONDITIONS ===")
    linear_results, round_means = test_linear_trends(df)
    
    # Visualize the results
    plot_contributions_by_round(round_means)
    print("\nAnalysis complete. Figure saved as 'contributions_by_condition.png'")
    
    return df, f_stat, p_value, linear_results, round_means

if __name__ == "__main__":
    results_dir = "results/03-25-2025_16-16"
    df, f_stat, p_value, linear_results, round_means = main(results_dir)