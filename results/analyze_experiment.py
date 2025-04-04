import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict

# Load the JSON data
results_dir = 'results/03-25-2025_16-16'
with open(f'{results_dir}/gossip_exp_0_gossip.json', 'r') as file:
    data = json.load(file)

# Extract relevant information
contributions = data['contributions']
earnings = data['earnings']
gossip_data = data['gossip']
player_ids = [f"Player_{i}" for i in range(1, 25)]
rounds = list(range(1, 7))  # 6 rounds total

# 1. Percentage of times the entire group chose to gossip when they could
total_opportunities = len(player_ids) * len(rounds)  # 24 players * 6 rounds
gossip_counts_by_round = {i: len(gossip_data.get(str(i), [])) for i in rounds}
total_gossips = sum(gossip_counts_by_round.values())

gossip_percentage = (total_gossips / total_opportunities) * 100
print(f"Percentage of times players chose to gossip when they could: {gossip_percentage:.2f}%")

# 2. Number of times each player chose to gossip
player_gossip_counts = Counter()
for round_gossips in gossip_data.values():
    for gossip in round_gossips:
        sender = gossip['sender']
        player_gossip_counts[sender] += 1

# 3. Average and std of contributions per round by player
player_contributions = defaultdict(list)
for round_num in rounds:
    round_contributions = contributions[str(round_num)]
    for player, contrib in round_contributions.items():
        player_contributions[player].append(contrib)

player_contrib_avg = {player: np.mean(contribs) for player, contribs in player_contributions.items()}
player_contrib_std = {player: np.std(contribs) for player, contribs in player_contributions.items()}

# 4. Average and std of earnings per round by player
player_earnings = defaultdict(list)
for round_num in rounds:
    round_earnings = earnings[str(round_num)]
    for player, earned in round_earnings.items():
        player_earnings[player].append(earned)

player_earnings_avg = {player: np.mean(earned) for player, earned in player_earnings.items()}
player_earnings_std = {player: np.std(earned) for player, earned in player_earnings.items()}

# Create plots
plt.figure(figsize=(15, 10))

# 1. Plot gossip percentage by round
plt.subplot(2, 2, 1)
gossip_pcts_by_round = [len(gossip_data.get(str(i), [])) / len(player_ids) * 100 for i in rounds]
plt.bar(rounds, gossip_pcts_by_round)
plt.title('Percentage of Players Who Gossiped Each Round')
plt.xlabel('Round')
plt.ylabel('Percentage (%)')
plt.ylim(0, 100)
plt.xticks(rounds)

# 2. Plot number of times each player chose to gossip
plt.subplot(2, 2, 2)
sorted_players = sorted(player_ids, key=lambda x: int(x.split('_')[1]))
gossip_counts = [player_gossip_counts[player] for player in sorted_players]
plt.bar(range(len(sorted_players)), gossip_counts)
plt.title('Number of Times Each Player Gossiped')
plt.xlabel('Player')
plt.ylabel('Count')
plt.xticks(range(len(sorted_players)), [f"P{p.split('_')[1]}" for p in sorted_players], rotation=90)

# 3. Plot average contributions with std error bars
plt.subplot(2, 2, 3)
players_by_contrib = sorted(player_ids, key=lambda x: player_contrib_avg[x])
contrib_avgs = [player_contrib_avg[player] for player in players_by_contrib]
contrib_stds = [player_contrib_std[player] for player in players_by_contrib]
plt.bar(range(len(players_by_contrib)), contrib_avgs, yerr=contrib_stds, capsize=4)
plt.title('Average Contributions by Player (with Std Dev)')
plt.xlabel('Player')
plt.ylabel('Contribution')
plt.xticks(range(len(players_by_contrib)), [f"P{p.split('_')[1]}" for p in players_by_contrib], rotation=90)
plt.axhline(y=np.mean(list(player_contrib_avg.values())), color='r', linestyle='--', label='Average')
plt.legend()

# 4. Plot average earnings with std error bars
plt.subplot(2, 2, 4)
players_by_earnings = sorted(player_ids, key=lambda x: player_earnings_avg[x])
earnings_avgs = [player_earnings_avg[player] for player in players_by_earnings]
earnings_stds = [player_earnings_std[player] for player in players_by_earnings]
plt.bar(range(len(players_by_earnings)), earnings_avgs, yerr=earnings_stds, capsize=4)
plt.title('Average Earnings by Player (with Std Dev)')
plt.xlabel('Player')
plt.ylabel('Earnings')
plt.xticks(range(len(players_by_earnings)), [f"P{p.split('_')[1]}" for p in players_by_earnings], rotation=90)
plt.axhline(y=np.mean(list(player_earnings_avg.values())), color='r', linestyle='--', label='Average')
plt.legend()

plt.tight_layout()
plt.savefig(f'{results_dir}/gossip_analysis.png', dpi=300)

# Additional plot: Average contributions by round
plt.figure(figsize=(10, 6))
round_avg_contributions = [np.mean(list(contributions[str(r)].values())) for r in rounds]
round_std_contributions = [np.std(list(contributions[str(r)].values())) for r in rounds]
plt.errorbar(rounds, round_avg_contributions, yerr=round_std_contributions, marker='o', capsize=5)
plt.title('Average Contributions by Round')
plt.xlabel('Round')
plt.ylabel('Contribution')
plt.xticks(rounds)
plt.ylim(0, 10)  # Maximum contribution is 10
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(f'{results_dir}/contributions_by_round.png', dpi=300)

print("Analysis complete. Results saved as gossip_analysis.png and contributions_by_round.png")

# Calculate additional statistics for insights
print("\nAdditional Statistics:")
print(f"Overall average contribution: {np.mean([c for round_contribs in contributions.values() for c in round_contribs.values()]):.2f}")
print(f"Overall average earnings: {np.mean([e for round_earnings in earnings.values() for e in round_earnings.values()]):.2f}")

# Correlation between gossip frequency and average contribution
gossip_freq = [player_gossip_counts[p] for p in player_ids]
avg_contribs = [player_contrib_avg[p] for p in player_ids]
corr = np.corrcoef(gossip_freq, avg_contribs)[0, 1]
print(f"Correlation between gossip frequency and average contribution: {corr:.2f}")

# How does contribution change from round 1 to round 6?
round1_avg = np.mean(list(contributions['1'].values()))
round6_avg = np.mean(list(contributions['6'].values()))
change_pct = ((round6_avg - round1_avg) / round1_avg) * 100
print(f"Change in average contribution from round 1 to round 6: {change_pct:.2f}%")