# statistical_analysis.py
import json
import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pathlib
import argparse

def load_data(results_dir):
    """Load all JSON files with experimental data across multiple experiments"""
    conditions = ['basic', 'gossip', 'gossip-with-ostracism']
    data = {}
    
    # Use glob to find all matching experiment files for each condition
    for condition in conditions:
        data[condition] = {
            'contributions': {},
            'earnings': {},
            'gossip': {},
            'ostracism': {}
        }
        
        # Find all files matching the pattern for this condition
        file_pattern = os.path.join(results_dir, f'gossip_exp_*_{condition}.json')
        result_files = list(pathlib.Path(results_dir).glob(f'gossip_exp_*_{condition}.json'))

        # Add reference components to check for consistency
        reference_components = None
        
        if not result_files:
            print(f"Warning: No files found for condition '{condition}'")
            continue
            
        print(f"Found {len(result_files)} files for condition '{condition}'")
        
        # Load and combine data from all experiments
        for file_path in result_files:
            try:
                with open(file_path, 'r') as f:
                    exp_data = json.load(f)
                
                # Check that all files use the same model architecture
                if 'config' in exp_data and 'components' in exp_data['config']:
                    if reference_components is None:
                        reference_components = exp_data['config']['components']
                        print(f"Reference components set from {file_path}")
                    else:
                        assert exp_data['config']['components'] == reference_components, \
                            f"File {file_path} has different components than the reference"
                    
                print(f"Successfully loaded {file_path}")
                
                # Combine contributions data
                if 'contributions' in exp_data:
                    for round_num, round_contribs in exp_data['contributions'].items():
                        if round_num not in data[condition]['contributions']:
                            data[condition]['contributions'][round_num] = {}
                        # Add player contributions for this experiment
                        for player, amount in round_contribs.items():
                            # Create a unique player ID by adding experiment identifier
                            exp_id = str(file_path).split('gossip_exp_')[1].split('_')[0]
                            unique_player = f"{player}_exp{exp_id}"
                            data[condition]['contributions'][round_num][unique_player] = amount
                
                # Combine earnings data similarly
                if 'earnings' in exp_data:
                    for round_num, round_earnings in exp_data['earnings'].items():
                        if round_num not in data[condition]['earnings']:
                            data[condition]['earnings'][round_num] = {}
                        for player, amount in round_earnings.items():
                            exp_id = str(file_path).split('gossip_exp_')[1].split('_')[0]
                            unique_player = f"{player}_exp{exp_id}"
                            data[condition]['earnings'][round_num][unique_player] = amount
                
                # Debug: Print structure info for the first file
                if file_path == result_files[0]:
                    print(f"Keys in {condition} data: {list(exp_data.keys())}")
                    if 'contributions' in exp_data:
                        rounds = list(exp_data['contributions'].keys())
                        print(f"Rounds in {condition}: {rounds[:3]}{'...' if len(rounds) > 3 else ''}")
                        if rounds:
                            first_round = rounds[0]
                            players = list(exp_data['contributions'][first_round].keys())
                            print(f"Players in round {first_round}: {players[:3]}{'...' if len(players) > 3 else ''}")
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
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

    # Calculate eta-squared for overall ANOVA
    # First, calculate sum of squares
    total_mean = wide_data.values.mean()
    ss_total = np.sum((wide_data.values - total_mean)**2)
    
    # Between-groups sum of squares
    ss_between = sum(len(wide_data[cond]) * (wide_data[cond].mean() - total_mean)**2 for cond in conditions)
    
    # Calculate eta-squared
    eta_squared = ss_between / ss_total
    
    print("\nANOVA Results (Between Conditions):")
    print(f"F({len(conditions)-1}, {len(wide_data)-len(conditions)}): F = {f_stat:.2f}, p = {p_value:.6f}, η² = {eta_squared:.2f}")
    
    # 2. Pairwise comparisons
    print("\nPairwise Comparisons:")
    pairwise_results = {}

    for i, cond1 in enumerate(conditions):
        for j, cond2 in enumerate(conditions):
            if j > i:  # Avoid duplicates
                # Calculate t-statistic for paired samples
                t_stat, p_val = stats.ttest_rel(wide_data[cond1], wide_data[cond2])

                # Calculate degrees of freedom for paired samples
                df = len(wide_data[cond1]) - 1

                # Calculate F-statistic (for paired samples, F = t²)
                f_pairwise = t_stat**2

                # Calculate eta-squared for paired comparison
                # For paired samples: η² = t² / (t² + df)
                eta_sq_pair = f_pairwise / (f_pairwise + df)

                # Store results
                pair_key = f"{cond1}_vs_{cond2}"
                pairwise_results[pair_key] = {
                    't': t_stat,
                    'p': p_val,
                    'F': f_pairwise,
                    'df': (1, df),
                    'eta_squared': eta_sq_pair
                }
                
                print(f"{cond1} vs {cond2}: F(1, {df}) = {f_pairwise:.2f}, p = {p_val:.6f}, η² = {eta_sq_pair:.2f}")
                print(f"  t({df}) = {t_stat:.2f}, p = {p_val:.6f}")
    
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

        # Calculate eta-squared for the linear trend
        # For a simple regression, eta-squared is equivalent to r-squared
        eta_squared = r_value**2
        
        # Save results
        results[condition] = {
            'slope': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'f_stat': f_stat,
            'f_p_value': f_p_value,
            'intercept': intercept,
            'std_err': std_err,
            'eta_squared': eta_squared
        }
        
        print(f"\nCondition: {condition}")
        print(f"Slope: {slope:.4f} (points per round)")
        print(f"F(1, {dof}) = {f_stat:.2f}, p = {p_value:.6f}, η² = {eta_squared:.2f}")
        
        if p_value < 0.05:
            direction = "increasing" if slope > 0 else "decreasing"
            print(f"There is a significant {direction} linear trend.")
        else:
            print("There is no significant linear trend.")
    
    # Compare linear trends between conditions
    print("\nComparison of Linear Trends Between Conditions:")
    trend_comparisons = {}
    
    for i, cond1 in enumerate(conditions):
        for j, cond2 in enumerate(conditions):
            if j > i:  # Avoid duplicates
                # Test for differences in slopes
                t_stat = (results[cond1]['slope'] - results[cond2]['slope']) / np.sqrt(
                    results[cond1]['std_err']**2 + results[cond2]['std_err']**2)
                # Calculate p-value from t-statistic (with appropriate degrees of freedom)
                dof = 2 * (6 - 2)  # 2 * (n - 2)
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), dof))
                
                # Calculate F-statistic for comparison
                f_comp = t_stat**2
                
                # Calculate eta-squared for the comparison
                eta_sq_comp = t_stat**2 / (t_stat**2 + dof)
                
                comparison_key = f"{cond1}_vs_{cond2}"
                trend_comparisons[comparison_key] = {
                    't': t_stat,
                    'p': p_val,
                    'F': f_comp,
                    'df': (1, dof),
                    'eta_squared': eta_sq_comp
                }
                
                print(f"{cond1} vs {cond2} (trend comparison):")
                print(f"  F(1, {dof}) = {f_comp:.2f}, p = {p_val:.6f}, η² = {eta_sq_comp:.2f}")
                print(f"  t({dof}) = {t_stat:.2f}, p = {p_val:.6f}")
    
    return results, round_means

def plot_contributions_by_round(round_means, results_dir):
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
    
    # use results_dir to save the figure
    plt.savefig(os.path.join(results_dir, 'contributions_by_condition.png'), dpi=300)
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
    plot_contributions_by_round(round_means, results_dir)
    print("\nAnalysis complete. Figure saved as 'contributions_by_condition.png'")
    
    return df, f_stat, p_value, linear_results, round_means

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Statistical analysis of gossip experiment results')
    parser.add_argument('--results_dir', type=str, required=True, 
                        help='Directory containing experiment result files')
    args = parser.parse_args()
    
    print(f"Analyzing results from: {args.results_dir}")
    df, f_stat, p_value, linear_results, round_means = main(args.results_dir)