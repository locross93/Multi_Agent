# main.py
import json
import os
from typing import Optional
import datetime

from concordia.language_model import gpt_model
from concordia.language_model import language_model
import sentence_transformers

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

def main():
    # Setup paths
    api_key_path = 'lc_api_key.json'
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    # Setup model and embedder
    model = setup_language_model(api_key_path=api_key_path)
    embedder = setup_embedder()

    # Run experiment
    from tpp_scenario import test_tpp_hypothesis
    test_tpp_hypothesis(model, embedder)

if __name__ == "__main__":
    main()