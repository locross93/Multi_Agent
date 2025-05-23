# main.py
import json
import os
from typing import Optional
import argparse
import pathlib
from datetime import datetime

from concordia.language_model import gpt_model
from concordia.language_model import language_model
import sentence_transformers

from tpp_scenario import test_tpp_hypothesis

# Load API key
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
        model_name='gpt-4o',  # or 'gpt-4-turbo-preview' for GPT-4 Turbo
    )
    # API_type = 'openai'
    # model = utils.language_model_setup(
    #     api_type=API_TYPE,
    #     model_name=MODEL_NAME,
    #     api_key=API_KEY,
    #     disable_language_model=DISABLE_LANGUAGE_MODEL,
    # )
    return model

def setup_embedder():
    """Setup sentence embedder."""
    _embedder_model = sentence_transformers.SentenceTransformer(
        'sentence-transformers/all-mpnet-base-v2')
    return lambda x: _embedder_model.encode(x, show_progress_bar=False)

def get_next_experiment_id(save_dir: pathlib.Path) -> int:
    """Find the highest experiment ID in the directory and return next available ID."""
    save_dir = pathlib.Path(save_dir)
    existing_files = list(save_dir.glob('tpp_exp_*.json'))
    if not existing_files:
        return 0
        
    # Extract experiment IDs from filenames
    exp_ids = []
    for f in existing_files:
        try:
            # Extract number between 'tpp_exp_' and '.json'
            exp_id = int(f.stem.split('tpp_exp_')[1])
            exp_ids.append(exp_id)
        except (IndexError, ValueError):
            continue
            
    return max(exp_ids) + 1 if exp_ids else 0

def main():
     # Set up argument parser
    parser = argparse.ArgumentParser(description='Run multiple TPP experiments')
    parser.add_argument('--save_dir', type=str, 
                       default=datetime.now().strftime('%m-%d-%Y_%H-%M'),
                       help='Directory name for saving results (default: current timestamp)')
    parser.add_argument('--n_experiments', type=int, default=5,
                       help='Number of experiments to run (default: 5)')
    parser.add_argument('--conditions', type=str, nargs='+', 
                       choices=['public', 'anonymous', 'all'],
                       default=['all'],
                       help='Conditions to run (choices: public, anonymous, all)')
    args = parser.parse_args()

    # Create results directory if it doesn't exist
    results_dir = os.path.join('results/tpp', args.save_dir)
    os.makedirs(results_dir, exist_ok=True)

    # Setup paths
    api_key_path = 'lc_api_key.json'

    # Setup model and embedder
    model = setup_language_model(api_key_path=api_key_path)
    embedder = setup_embedder()

    # Get starting experiment ID
    start_id = get_next_experiment_id(results_dir)
    print(f"Starting with experiment ID: {start_id}")

    # Process the conditions argument
    if args.conditions == 'all':
        args.conditions = ['public', 'anonymous']
    # assert args.conditions is a list
    assert isinstance(args.conditions, list)

    # Run N experiments
    for i in range(args.n_experiments):
        current_id = start_id + i
        print(f"\nRunning experiment {i+1}/{args.n_experiments} (ID: {current_id})")
        test_tpp_hypothesis(
            model=model,
            embedder=embedder,
            save_dir=results_dir,
            experiment_id=current_id,
            conditions=args.conditions,
        )

if __name__ == "__main__":
    main()