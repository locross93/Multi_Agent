# main.py
import json
import os
import argparse
import pathlib
from datetime import datetime
import random
import asyncio
from typing import Dict, List, Optional, Tuple

from concordia.language_model import gpt_model
from concordia.language_model import language_model
import sentence_transformers

from gossip_scenario_async import test_gossip_ostracism_hypothesis_async
import parse_agent_logs
from results import plot_gossip_results

# Modified main function
async def main_async():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run Gossip and Ostracism experiments')
    parser.add_argument('--save_dir', type=str, 
                       default=datetime.now().strftime('%m-%d-%Y_%H-%M'),
                       help='Directory name for saving results (default: current timestamp)')
    parser.add_argument('--n_experiments', type=int, default=1,
                       help='Number of experiments to run (default: 1)')
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2],
                       help='Validation stage (1: validate components, 2: novel predictions)')
    parser.add_argument('--num_rounds', type=int, default=6,
                       help='Number of rounds to run (default: 6)')
    parser.add_argument('--num_players', type=int, default=24,
                       help='Number of players to run (default: 24)')
    parser.add_argument('--conditions', type=str, nargs='+', 
                       choices=['gossip-with-ostracism', 'gossip', 'basic', 'all'],
                       default=['all'],
                       help='Conditions to run (choices: gossip-with-ostracism, gossip, basic, all)')
    args = parser.parse_args()

    # Process the conditions argument
    if 'all' in args.conditions or args.conditions == ['all']:
        args.conditions = ['gossip-with-ostracism', 'gossip', 'basic']
    # assert args.conditions is a list
    assert isinstance(args.conditions, list)

    # Create results directory if it doesn't exist
    results_dir = os.path.join('results', args.save_dir)
    os.makedirs(results_dir, exist_ok=True)

    # Setup paths
    api_key_path = 'lc_api_key.json'

    # Setup model and embedder
    model = setup_language_model(api_key_path=api_key_path)
    embedder = setup_embedder()

    # Get starting experiment ID
    start_id = get_next_experiment_id(results_dir)
    print(f"Starting with experiment ID: {start_id}")

    # Run N experiments
    for i in range(args.n_experiments):
        current_id = start_id + i
        print(f"\nRunning experiment {i+1}/{args.n_experiments} (ID: {current_id})")
        await test_gossip_ostracism_hypothesis_async(
            model=model,
            embedder=embedder,
            save_dir=results_dir,
            experiment_id=current_id,
            validation_stage=args.stage,
            num_rounds=args.num_rounds,
            num_players=args.num_players,
            conditions=args.conditions
        )

    return results_dir

def main():
    # Use asyncio.run to run our async main function
    results_dir = asyncio.run(main_async())

    # run parse_agent_logs.py 
    parse_agent_logs.process_agent_logs(results_dir)

    # run results/plot_gossip_results.py
    plot_gossip_results.plot_gossip_results(results_dir)

# Needed utility functions from the original code
def load_api_key(api_key_path: str) -> str:
    """Load API key from json file."""
    with open(api_key_path, 'r') as f:
        keys = json.load(f)
    return keys['API_KEY']

def setup_language_model(api_key_path: Optional[str] = None, api_key: Optional[str] = None) -> language_model.LanguageModel:
    """Setup language model with API key."""
    if api_key_path:
        api_key = load_api_key(api_key_path)
    elif not api_key:
        raise ValueError("Must provide either api_key_path or api_key")
        
    # Initialize GPT-4 model
    model = gpt_model.GptLanguageModel(
        api_key=api_key,
        model_name='gpt-4o',
    )
    return model

def setup_embedder():
    """Setup sentence embedder."""
    _embedder_model = sentence_transformers.SentenceTransformer(
        'sentence-transformers/all-mpnet-base-v2')
    return lambda x: _embedder_model.encode(x, show_progress_bar=False)

def get_next_experiment_id(save_dir: pathlib.Path) -> int:
    """Find the highest experiment ID in the directory and return next available ID."""
    save_dir = pathlib.Path(save_dir)
    existing_files = list(save_dir.glob('gossip_exp_*.json'))
    if not existing_files:
        return 0
        
    # Extract experiment IDs from filenames
    exp_ids = []
    for f in existing_files:
        try:
            exp_id = int(f.stem.split('gossip_exp_')[1])
            exp_ids.append(exp_id)
        except (IndexError, ValueError):
            continue
            
    return max(exp_ids) + 1 if exp_ids else 0

if __name__ == "__main__":
    main()