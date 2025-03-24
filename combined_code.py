
################################################################################
# FILE: main_gossip.py
################################################################################

# main.py
import json
import os
import argparse
import pathlib
from datetime import datetime
import random
from typing import Dict, List, Optional, Tuple

from concordia.language_model import gpt_model
from concordia.language_model import language_model
import sentence_transformers

from gossip_scenario import test_gossip_ostracism_hypothesis

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

def main():
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
    args = parser.parse_args()

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
        test_gossip_ostracism_hypothesis(
            model=model,
            embedder=embedder,
            save_dir=results_dir,
            experiment_id=current_id,
            validation_stage=args.stage,
            num_rounds=args.num_rounds,
            num_players=args.num_players
        )

if __name__ == "__main__":
    main()

################################################################################
# FILE: gossip_scenario.py
################################################################################

# gossip_scenario.py
from dataclasses import dataclass
import os
import re
import datetime
import json
import pathlib
import random
from typing import Dict, List, Optional, Tuple, Callable, Union
import numpy as np

from concordia.agents import entity_agent_with_logging
from concordia.clocks import game_clock
from concordia.language_model import language_model
from concordia.utils import measurements as measurements_lib
from concordia.environment import game_master
from concordia.associative_memory import associative_memory, formative_memories
from concordia.memory_bank import legacy_associative_memory 
from concordia.components.agent import memory_component
from concordia.typing.entity import free_action_spec, choice_action_spec, float_action_spec

from agent_components_gossip import build_gossip_agent
from agent_logging import (
    setup_agent_logging, 
    create_logger_for_agent, 
    add_logging_to_agent,
    add_logging_to_experiment
)
from personas_gossip import Persona, PERSONAS, assign_personas

# Constants for the public goods game
PUBLIC_GOODS_ENDOWMENT = 10.0
PUBLIC_GOODS_MULTIPLIER = 2.0
OSTRACISM_MULTIPLIER = 1.5  # Reduced multiplier when someone is ostracized

# Action spec for public goods contribution
CONTRIBUTION_ACTION_SPEC_COT = free_action_spec(
    call_to_action=(
        "As {name}, you have ${endowment} to allocate in the public goods game. "
        "How much money you will contribute to the group fund (any amount between $0 and ${endowment})? "
        "The group fund will be multiplied by {multiplier} and divided equally among all players. "
        "Any money not contributed stays in your private account. "
        "Reflect on your decision step by step given all of the information in this prompt and your previous observations. Think through your reasoning carefully. "
        "After your reasoning, end with your final decision using this exact format: "
        "===DECISION=== "
        "I choose to contribute $X to the group fund."
    ),
    tag="contribution_action"
)

CONTRIBUTION_ACTION_SPEC = free_action_spec(
    call_to_action=(
        "As {name}, you have ${endowment} to allocate in the public goods game. "
        "How much money you will contribute to the group fund (any amount between $0 and ${endowment})? "
        "The group fund will be multiplied by {multiplier} and divided equally among all players. "
        "Any money not contributed stays in your private account. "
        "Think step by step and make your final decision using this exact format: "
        "I choose to contribute $X to the group fund."
    ),
    tag="contribution_action"
)

# Action spec for gossip decision
GOSSIP_ACTION_SPEC_COT = free_action_spec(
    call_to_action=(
        "You've just played a round with {players}. "
        "Would you like to send a note about one of them to their future interaction partners? "
        "If yes, clearly state which player # and what you want to say about them. "
        "If you don't want to send a note, just say so. "
        "For example: 'I want to send a note about Player_10: They are selfish and only contributed $2 which was much less than everyone else.' "
        "Or: 'I choose not to send any notes this round.' "
        "Step 1: Reflect on your decision step by step given all of the information in this prompt and your previous observations. Think through your reasoning carefully. "
        "Step 2: After your reasoning, end with your decision using this exact format: "
        "===DECISION=== "
        "If sending a note: \"I send a note about Player_[X]: [your message]\" "
        "If not sending: \"I choose not to send any notes this round.\""
    ),
    tag="gossip_action"
)

GOSSIP_ACTION_SPEC = free_action_spec(
    call_to_action=(
        "You've just played a round with {players}. "
        "Would you like to send a note about one of them to their future interaction partners? "
        "If yes, clearly state which player # and what you want to say about them. "
        "If you don't want to send a note, just say so. "
        "For example: 'I want to send a note about Player_10: They are selfish and only contributed $2 which was much less than everyone else.' "
        "Or: 'I choose not to send any notes this round.'"
    ),
    tag="gossip_action"
)

# Action spec for ostracism decision (converting to free_action_spec)
OSTRACISM_ACTION_SPEC_COT = free_action_spec(
    call_to_action=(
        "You are about to play with {players}. "
        "Based on the notes you received, would you like to exclude one person from your group? "
        "If yes, clearly state which player # and why you want to exclude them. "
        "If you don't want to exclude anyone, just say so. "
        "If at least 2 people vote to exclude someone, they will be excluded for the next round. "
        "Note that if someone is excluded, the group fund multiplier decreases from 2.0 to 1.5. "
        "Step 1: Reflect on your decision step by step given all of the information in this prompt and your previous observations. Think through your reasoning carefully. "
        "Step 2: After your reasoning, end with your decision using this exact format: "
        "===DECISION=== "
        "I choose to exclude Player_[X] from my group. "
        "Or: \"I choose not to exclude anyone from my group.\""
    ),
    tag="ostracism_action"
)

OSTRACISM_ACTION_SPEC = free_action_spec(
    call_to_action=(
        "You are about to play with {players}. "
        "Based on the notes you received, would you like to exclude one person from your group? "
        "If yes, clearly state which player # and why you want to exclude them. "
        "If you don't want to exclude anyone, just say so. "
        "If at least 2 people vote to exclude someone, they will be excluded for the next round. "
        "Note that if someone is excluded, the group fund multiplier decreases from 2.0 to 1.5. "
        "Make your decision using this exact format: "
        "I choose to exclude Player_[X] from my group. "
        "Or: \"I choose not to exclude anyone from my group.\""
    ),
    tag="ostracism_action"
)

@dataclass
class GossipScenarioConfig:
    """Configuration for Gossip experiment scenario."""
    endowment: float = PUBLIC_GOODS_ENDOWMENT
    multiplier: float = PUBLIC_GOODS_MULTIPLIER
    ostracism_multiplier: float = OSTRACISM_MULTIPLIER
    save_dir: str = None
    experiment_id: int = None
    condition: str = "basic"  # "basic", "gossip", or "gossip-with-ostracism"
    scenario_description: str = """You are participating in an economic game with 24 participants.
    
    Everyone will be randomly assigned to groups of 4 for each round. You will play a total of 6 rounds, and 
    in each round you will be in a different group with people you haven't played with before.
    
    In each round, all players receive $10. Each player can contribute any amount between $0 and $10 to a group fund.
    Any amount contributed to the group fund will be multiplied by 2 and divided equally among all 4 group members.
    Any amount not contributed stays in your private account.
    
    For example, if everyone contributes $10, the group fund becomes $40, which is multiplied to $80, and each person 
    receives $20 (a $10 profit). However, if you contribute nothing while everyone else contributes $10, the group fund
    becomes $30, which is multiplied to $60, and each person receives $15. So you would have $25 total ($10 kept + $15 from group).
    """


class GossipGameMaster:
    """Game master for Gossip experiment."""

    def __init__(
        self,
        config: GossipScenarioConfig,
        model: language_model.LanguageModel,
        embedder,
        clock: game_clock.MultiIntervalClock,
        measurements: measurements_lib.Measurements,
        agents: List[entity_agent_with_logging.EntityAgentWithLogging],
        condition: str,
    ):
        # Create memory for game master
        self.memory = associative_memory.AssociativeMemory(embedder)
        
        # Store all parameters directly
        self.model = model
        self.agents = agents  # Store agents as a standalone attribute
        self.condition = condition
        self.config = config
        self.clock = clock
        self.measurements = measurements
        self.round = 0
        self.results_log = []
        
        # Track groupings, contributions, and earnings
        self.groups = {}
        self.contributions = {}
        self.earnings = {}
        self.gossip_messages = {}
        self.ostracism_votes = {}
        self.ostracized_players = set()
        
        # Generate player groupings for all rounds in advance
        self.generate_round_robin_groups(agents)

    def generate_round_robin_groups(self, agents: List[entity_agent_with_logging.EntityAgentWithLogging]):
        """Generate round-robin groupings to ensure no two players are paired more than once."""
        n_players = len(agents)
        n_rounds = 6
        
        player_names = [agent.name for agent in agents]
        
        # Store predetermined groupings for each round
        self.round_groupings = {}
        
        for round_num in range(1, n_rounds + 1):
            # Shuffle players for this round
            shuffled_players = player_names.copy()
            random.shuffle(shuffled_players)
            
            # Form groups of 4
            groups = {}
            for i in range(0, n_players, 4):
                group_id = f"Group_{i//4 + 1}"
                group_members = shuffled_players[i:i+4]
                groups[group_id] = group_members
            
            self.round_groupings[round_num] = groups

    def get_player_by_name(self, name: str) -> entity_agent_with_logging.EntityAgentWithLogging:
        """Helper method to get player by name."""
        for player in self.agents:  # Use self.agents instead of self.players
            if player.name == name:
                return player
        raise ValueError(f"No player found with name: {name}")

    def run_round(self) -> Dict:
        """Run a single round of the public goods game."""
        self.round += 1
        print(f"Running round {self.round}...")
        
        # Initialize data structures for this round
        self.contributions[self.round] = {}
        self.earnings[self.round] = {}
        self.groups[self.round] = self.round_groupings[self.round]
        
        # Process ostracism from previous round (if applicable)
        if self.condition == "gossip-with-ostracism" and self.round > 1:
            # Update groups to exclude ostracized players
            self.update_groups_for_ostracism()
            
        # 1. Inform players of their current group and condition
        for group_id, members in self.groups[self.round].items():
            for player_name in members:
                agent = self.get_player_by_name(player_name)
                group_info = f"Round {self.round}: You are in {group_id} with {', '.join([m for m in members if m != player_name])}."
                agent.observe(group_info)
                if self.condition == "gossip-with-ostracism":
                    condition_info = (
                        f"You are in the gossip-with-ostracism condition, which means that after each round, "
                        f"players will be able to send a note about one of their group members to their future "
                        f"interaction partners. Before the start of the next round, you will receive notes from "
                        f"the previous round, and can vote to exclude one of your group members based on the "
                        f"gossip you've received about them."
                    )
                    agent.observe(condition_info)
                elif self.condition == "gossip":
                    condition_info = f"You are in the gossip condition, which means that after each round, players will be able to send a note about one of their group members to their future interaction partners."
                    agent.observe(condition_info)

        # 2. Receive gossip (if applicable)
        if self.condition in ["gossip", "gossip-with-ostracism"] and self.round > 1:
            self.deliver_gossip()
        
        # 3. Vote for ostracism (if applicable)
        if self.condition == "gossip-with-ostracism" and self.round > 1:
            self.conduct_ostracism_vote()
        
        # 4. Collect contributions
        for group_id, members in self.groups[self.round].items():
            for player_name in members:
                if player_name not in self.ostracized_players:
                    agent = self.get_player_by_name(player_name)
                    multiplier = self.config.multiplier
                    
                    # If someone in the group is ostracized, use reduced multiplier
                    if self.condition == "gossip-with-ostracism" and len(set(members) & self.ostracized_players) > 0:
                        multiplier = self.config.ostracism_multiplier
                    
                    contribution_spec = free_action_spec(
                        call_to_action=CONTRIBUTION_ACTION_SPEC.call_to_action.format(
                            name=agent.name,
                            endowment=self.config.endowment,
                            multiplier=multiplier
                        ),
                        tag="contribution_action"
                    )
                    
                    contribution_action = agent.act(contribution_spec)
                    match = re.search(r'\$(\d+\.?\d*)', contribution_action)
                    contribution = float(match.group(1)) if match else 0.0
                    
                    # Ensure contribution is within bounds
                    assert contribution >= 0.0
                    assert contribution <= self.config.endowment
                    
                    if contribution == None:
                        breakpoint()
                    self.contributions[self.round][player_name] = contribution
                    
                    # Log the contribution
                    contribution_event = f"{player_name} contributes ${contribution:.1f} to the group fund."
                    self.results_log.append({
                        'round': self.round,
                        'time': str(self.clock.now()),
                        'group': group_id,
                        'player': player_name,
                        'action': contribution_action,
                        'event': contribution_event,
                        'contribution': contribution,
                        'condition': self.condition
                    })
                    
                    # # Have all group members observe the contribution
                    # for member_name in members:
                    #     if member_name != player_name and member_name not in self.ostracized_players:
                    #         member = self.get_player_by_name(member_name)
                    #         member.observe(contribution_event)
                else:
                    # Ostracized players contribute 0
                    self.contributions[self.round][player_name] = 0.0

        # Inform all players about contributions after all decisions are made
        for group_id, members in self.groups[self.round].items():
            active_members = [m for m in members if m not in self.ostracized_players]
            for player_name in active_members:
                agent = self.get_player_by_name(player_name)
                for member_name in active_members:
                    if member_name != player_name:
                        contribution = self.contributions[self.round].get(member_name, 0.0)
                        contribution_event = f"{member_name} contributes ${contribution:.1f} to the group fund."
                        agent.observe(contribution_event)

        # 5. Calculate and distribute earnings
        for group_id, members in self.groups[self.round].items():
            active_members = [m for m in members if m not in self.ostracized_players]
            
            if active_members:  # Ensure there are active members in the group
                # Calculate total contribution for the group
                total_contribution = sum(self.contributions[self.round].get(m, 0.0) for m in active_members)
                
                # Apply multiplier
                multiplier = self.config.multiplier
                if self.condition == "gossip-with-ostracism" and len(set(members) & self.ostracized_players) > 0:
                    multiplier = self.config.ostracism_multiplier
                
                # Calculate group return
                group_return = total_contribution * multiplier / len(active_members)
                
                # Distribute earnings
                for member_name in active_members:
                    personal_contribution = self.contributions[self.round].get(member_name, 0.0)
                    earnings = self.config.endowment - personal_contribution + group_return
                    self.earnings[self.round][member_name] = earnings
                    
                    # Inform player of the results
                    agent = self.get_player_by_name(member_name)
                    results_msg = (
                        f"Round {self.round} Results: "
                        f"Total group contribution: ${total_contribution:.1f}, "
                        f"Your contribution: ${personal_contribution:.1f}, "
                        f"Your earnings: ${earnings:.1f} "
                        f"(${self.config.endowment - personal_contribution:.1f} kept + ${group_return:.1f} from group fund)"
                    )
                    agent.observe(results_msg)
            
            # Set earnings to 0 for ostracized players
            for member_name in members:
                if member_name in self.ostracized_players:
                    self.earnings[self.round][member_name] = 0.0
                    agent = self.get_player_by_name(member_name)
                    agent.observe(f"Round {self.round}: You were excluded from the group and earned $0.")
        
        # 6. Allow gossip (if applicable)
        if self.condition in ["gossip", "gossip-with-ostracism"]:
            self.collect_gossip()
        
        # Return round data
        round_data = {
            'round': self.round,
            'groups': self.groups[self.round],
            'contributions': self.contributions[self.round],
            'earnings': self.earnings[self.round],
            'gossip': self.gossip_messages.get(self.round, []),
            'ostracized': list(self.ostracized_players)
        }

        return round_data

    def update_groups_for_ostracism(self):
        """Update group assignments to handle ostracized players."""
        # Ostracized players remain assigned to their groups but don't participate
        # This is handled during contribution collection
        pass

    def deliver_gossip(self):
        """Deliver gossip messages from previous round to recipients."""
        if self.round - 1 not in self.gossip_messages:
            return
            
        for gossip in self.gossip_messages[self.round - 1]:
            for recipient_name in gossip['recipients']:
                recipient = self.get_player_by_name(recipient_name)
                gossip_msg = f"Note about {gossip['target']}: {gossip['message']}"
                recipient.observe(gossip_msg)

    def conduct_ostracism_vote(self):
        """Conduct ostracism voting."""
        if self.condition != "gossip-with-ostracism":
            return
            
        self.ostracism_votes[self.round] = {}
        
        # Have each agent vote
        for group_id, members in self.groups[self.round].items():
            # Create a mapping of player names to letters for the voting
            player_mapping = {}
            for i, player_name in enumerate(members):
                letter = chr(65 + i)  # A, B, C, D
                player_mapping[f"Player {letter}"] = player_name
                
            # Create the reverse mapping for displaying options
            letter_mapping = {v: k for k, v in player_mapping.items()}
            
            for voter_name in members:
                agent = self.get_player_by_name(voter_name)
                
                # Format the options for this specific group
                options = ["No, I don't want to exclude anyone"]
                for member_name in members:
                    if member_name != voter_name:
                        options.append(f"Yes, I vote to exclude {letter_mapping[member_name]}")
                
                # Create a custom action spec for this group
                ostracism_spec = choice_action_spec(
                    call_to_action=OSTRACISM_ACTION_SPEC.call_to_action.format(
                        players=", ".join([letter_mapping[m] for m in members if m != voter_name])
                    ),
                    options=options,
                    tag="ostracism_action"
                )

                decision_text = agent.act(ostracism_spec)
                # Check if the agent chose not to exclude anyone
                if "not to exclude" in decision_text.lower() or "choose not to exclude" in decision_text.lower():
                    # No exclusion vote
                    pass
                else:
                    # Look for Player_X pattern
                    match = re.search(r'Player_(\d+)', decision_text)
                    if match:
                        player_number = match.group(1)
                        target_player = f"Player_{player_number}"
                        
                        # Map the player number to the actual player name
                        for actual_name, display_name in player_mapping.items():
                            if display_name == target_player:
                                target_name = actual_name
                                
                                # Record the vote
                                if target_name not in self.ostracism_votes[self.round]:
                                    self.ostracism_votes[self.round][target_name] = []
                                self.ostracism_votes[self.round][target_name].append(voter_name)
                                
                                # Log the vote
                                vote_event = f"{voter_name} votes to exclude {target_name}."
                                self.results_log.append({
                                    'round': self.round,
                                    'time': str(self.clock.now()),
                                    'player': voter_name,
                                    'action': vote_action,
                                    'event': vote_event,
                                    'condition': self.condition
                                })
        
        # Determine who gets ostracized (2+ votes required)
        self.ostracized_players.clear()
        for target_name, voters in self.ostracism_votes[self.round].items():
            if len(voters) >= 2:
                self.ostracized_players.add(target_name)
                
                # Inform the ostracized player
                target = self.get_player_by_name(target_name)
                target.observe(f"Round {self.round}: You have been excluded from your group for this round.")
                
                # Inform the group
                group_id = next(g for g, members in self.groups[self.round].items() if target_name in members)
                for member_name in self.groups[self.round][group_id]:
                    if member_name != target_name:
                        member = self.get_player_by_name(member_name)
                        member.observe(f"Round {self.round}: {target_name} has been excluded from your group for this round.")

    def collect_gossip(self):
        """Collect gossip from agents about their current group members."""
        if self.condition not in ["gossip", "gossip-with-ostracism"]:
            return
            
        self.gossip_messages[self.round] = []
        
        # Determine the next round's groupings
        next_round = self.round + 1
        if next_round > max(self.round_groupings.keys()):
            return  # No more rounds
            
        next_groups = self.round_groupings[next_round]
        
        # For each player, collect gossip
        for group_id, members in self.groups[self.round].items():
            for sender_name in members:
                if sender_name not in self.ostracized_players:
                    agent = self.get_player_by_name(sender_name)
                    
                    # Create a mapping of player names to letters for the gossip
                    player_mapping = {}
                    for i, player_name in enumerate(members):
                        if player_name != sender_name:
                            letter = chr(65 + i)  # A, B, C, D
                            player_mapping[f"Player {letter}"] = player_name
                    
                    # Create the reverse mapping for displaying options
                    letter_mapping = {v: k for k, v in player_mapping.items()}
                    
                    # Format the options for this specific group
                    options = ["No, I choose not to send any notes"]
                    for member_name in members:
                        if member_name != sender_name:
                            options.append(f"Yes, I will send a note about {letter_mapping[member_name]}: (describe what you want to say)")
                    
                    # Create a custom action spec for this group
                    gossip_spec = free_action_spec(
                        call_to_action=GOSSIP_ACTION_SPEC.call_to_action.format(
                            players=", ".join([letter_mapping[m] for m in members if m != sender_name])
                        ),
                        tag="gossip_action"
                    )

                    gossip_action = agent.act(gossip_spec)
                    # Look for the decision delimiter
                    decision_text = gossip_action
                    
                    # Check if the agent chose not to gossip
                    if "not to send" in decision_text.lower() or "choose not to" in decision_text.lower():
                        continue
                    
                    # Look for Player_X pattern
                    match = re.search(r'Player_(\d+)', decision_text)
                    if match:
                        player_number = match.group(1)
                        target_player = f"Player_{player_number}"
                        
                        # Map the player number to the actual player name
                        for actual_name, display_name in player_mapping.items():
                            if display_name == target_player:
                                target_name = actual_name
                                
                                # Extract the message after the player identifier
                                message_match = re.search(f"Player_{player_number}[:]?\s*(.*)", decision_text, re.IGNORECASE)
                                if message_match:
                                    message = message_match.group(1).strip()
                                    if len(message) > 0:
                                        # Find the target's next group members (recipients of the gossip)
                                        recipients = []
                                        for next_group_id, next_members in next_groups.items():
                                            if target_name in next_members:
                                                recipients = [m for m in next_members if m != target_name]
                                                break
                                        
                                        # Record the gossip
                                        if recipients:
                                            self.gossip_messages[self.round].append({
                                                'sender': sender_name,
                                                'target': target_name,
                                                'recipients': recipients,
                                                'message': message
                                            })
                                            
                                            # Log the gossip
                                            gossip_event = f"{sender_name} sends gossip about {target_name} to their future group members: '{message}'"
                                            self.results_log.append({
                                                'round': self.round,
                                                'time': str(self.clock.now()),
                                                'player': sender_name,
                                                'action': gossip_action,
                                                'event': gossip_event,
                                                'condition': self.condition
                                            })
                    

    def save_results(self):
        """Save results to JSON file."""
        results = {
            'config': self.config.__dict__,
            'condition': self.condition,
            'groups': {str(k): v for k, v in self.groups.items()},
            'contributions': {str(k): v for k, v in self.contributions.items()},
            'earnings': {str(k): v for k, v in self.earnings.items()},
            'gossip': {str(k): v for k, v in self.gossip_messages.items() if k in self.gossip_messages},
            'ostracism': {str(k): v for k, v in self.ostracism_votes.items() if k in self.ostracism_votes},
            'events': self.results_log
        }
        
        # Create results directory if it doesn't exist
        results_file = os.path.join(self.config.save_dir, f'gossip_exp_{self.config.experiment_id}_{self.condition}.json')
        
        # Save results to file
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

def run_gossip_experiment(
    model: language_model.LanguageModel,
    embedder,
    clock: game_clock.MultiIntervalClock,
    measurements: measurements_lib.Measurements,
    config: GossipScenarioConfig,
    condition: str,
    agents: List[entity_agent_with_logging.EntityAgentWithLogging],
    num_rounds: int = 6,
) -> None:
    """Run gossip experiment with logging."""

    # Setup agent logging
    log_dir = setup_agent_logging(config.save_dir, config.experiment_id, condition)

    # Add comprehensive logging to all agents and measurements
    agent_loggers = add_logging_to_experiment(agents, measurements, log_dir)

    # Create game master with agents
    gm = GossipGameMaster(
        config=config,
        model=model,
        embedder=embedder,
        clock=clock,
        measurements=measurements,
        agents=agents,
        condition=condition,
    )

    # Initialize agents with scenario context
    for agent in agents:
        if condition == "gossip":
            condition_desc = (
                             f"You are in the gossip condition, which means that after each round, "
                             f"you will have the opportunity to send a note about one of your group members to that "
                             f"person's future group. Before each round, you will see any notes sent about your upcoming "
                             f"group members"
            )
        elif condition == "gossip-with-ostracism":
            condition_desc = (
                             f"You are in the gossip-with-ostracism condition, which means that after each round, "
                             f"players will be able to send a note about one of their group members to their future "
                             f"interaction partners. Before the start of the next round, you will receive notes from "
                             f"the previous round, and can vote to exclude one of your group members based on the "
                             f"gossip you've received about them."
            )
        else:
            condition_desc = ""
            
        task_description = config.scenario_description + condition_desc
        agent.observe(task_description)
        # memory_component = agent.get_component('__memory__')
        # if memory_component:
        #     memory_component.add(task_description)

    # Run six rounds of the game
    for _ in range(num_rounds):
        gm.run_round()
        # Advance the clock
        clock.advance()

    # Save results
    gm.save_results()

def test_gossip_ostracism_hypothesis(
    model: language_model.LanguageModel,
    embedder: Callable[[Union[str, List[str]]], np.ndarray],
    save_dir: str,
    experiment_id: int,
    validation_stage: int = 1,
    num_rounds: int = 6,
    num_players: int = 24,
):
    """Run complete gossip-ostracism experiment testing various hypotheses."""
    
    # Initialize clock
    clock = game_clock.MultiIntervalClock(
        start=datetime.datetime.now(),
        step_sizes=[datetime.timedelta(minutes=5)],
    )
    
    # Initialize measurements
    measurements = measurements_lib.Measurements()

    # # Add raw data debugging
    # measurements.get_channel('ContributionDecision').subscribe(
    #     lambda data: print(f"DEBUG RAW DATA: {data}")
    # )

    def print_measurement(data):
        print(f"\n*** MEASUREMENT DATA: {data} ***\n") 

    measurements.get_channel('ContributionDecision').subscribe(print_measurement)
    measurements.get_channel('PersonalityReflection').subscribe(print_measurement)
    
    # STAGE 1: MODEL VALIDATION
    if validation_stage == 1:
        print("STAGE 1: MODEL VALIDATION - Testing which components are necessary")
        
        # Test Full Agents (personas and theory of mind)
        print("\nTesting full agents (personas and theory of mind)...")
        personas = assign_personas(n=num_players)
        
        # Run all three conditions with full agents
        for condition in ["gossip-with-ostracism", "gossip", "basic"]:
            full_agents = []
            for i in range(num_players):
                agent = build_gossip_agent(
                    config=formative_memories.AgentConfig(
                        name=f"Player_{i+1}",
                        goal="Participate in the public goods game",
                        extras={'main_character': True}
                    ),
                    model=model,
                    memory=associative_memory.AssociativeMemory(embedder),
                    clock=clock,
                    has_persona=True,
                    has_theory_of_mind=True,
                    persona=personas[i],
                )
                full_agents.append(agent)
            config = GossipScenarioConfig(
                save_dir=save_dir, 
                experiment_id=experiment_id,
                condition=condition
            )
            run_gossip_experiment(
                model=model,
                embedder=embedder,
                clock=clock,
                measurements=measurements,
                config=config,
                condition=condition,
                agents=full_agents,
                num_rounds=num_rounds,
            )
            
    # STAGE 2: NOVEL PREDICTION GENERATION
    else:
        print("STAGE 2: NOVEL PREDICTION GENERATION - Testing novel hypotheses")
        
        # Create full agents with personas and theory of mind for novel predictions
        personas = assign_personas(n=24)
        full_agents = []
        for i in range(24):
            agent = build_gossip_agent(
                config=formative_memories.AgentConfig(
                    name=f"Player_{i+1}",
                    goal="Participate in the public goods game",
                    extras={'main_character': True}
                ),
                model=model,
                memory=associative_memory.AssociativeMemory(embedder),
                clock=clock,
                has_persona=True,
                has_theory_of_mind=True,
                persona=personas[i],
            )
            full_agents.append(agent)
        
        # NOVEL HYPOTHESIS 1: Effect of inaccurate gossip
        # This would require modifications to the gossip mechanism to introduce inaccuracies
        
        # NOVEL HYPOTHESIS 2: Effect of public vs. private gossip
        # In the original experiment, gossip is sent privately to future partners
        # What if gossip was public (visible to all players)?
        
        # NOVEL HYPOTHESIS 3: Effect of temporary vs. permanent ostracism
        # In the original experiment, ostracism lasts for one round
        # What if ostracism was permanent (excluded for the rest of the game)?
        
        # For now, we'll just run the baseline conditions
        for condition in ["basic", "gossip", "gossip-with-ostracism"]:
            config = GossipScenarioConfig(
                save_dir=save_dir, 
                experiment_id=experiment_id,
                condition=condition
            )
            run_gossip_experiment(
                model=model,
                embedder=embedder,
                clock=clock,
                measurements=measurements,
                config=config,
                condition=condition,
                agents=full_agents,
            )

def analyze_results(save_dir: str):
    """Analyze and compare results between conditions."""
    # Load results for all conditions
    results_files = list(pathlib.Path(save_dir).glob('gossip_exp_*.json'))
    
    results_by_condition = {}
    for file_path in results_files:
        with open(file_path, 'r') as f:
            result = json.load(f)
            condition = result['condition']
            if condition not in results_by_condition:
                results_by_condition[condition] = []
            results_by_condition[condition].append(result)
    
    # Calculate average contributions per round by condition
    avg_contributions = {}
    for condition, results_list in results_by_condition.items():
        avg_contributions[condition] = []
        for round_num in range(1, 7):
            round_key = str(round_num)
            round_contribs = []
            for result in results_list:
                if round_key in result['contributions']:
                    round_contribs.extend(list(result['contributions'][round_key].values()))
            if round_contribs:
                avg_contributions[condition].append(sum(round_contribs) / len(round_contribs))
            else:
                avg_contributions[condition].append(0)
    
    # Print average contributions by condition and round
    print("\nAverage Contributions by Condition and Round:")
    for condition, avgs in avg_contributions.items():
        print(f"{condition}:")
        for round_num, avg in enumerate(avgs, 1):
            print(f"  Round {round_num}: ${avg:.2f}")
        print(f"  Overall: ${sum(avgs)/len(avgs):.2f}")
    
    # Analyze gossip content
    if any('gossip' in c for c in results_by_condition.keys()):
        print("\nGossip Analysis:")
        for condition in [c for c in results_by_condition.keys() if 'gossip' in c]:
            gossip_count = 0
            negative_gossip = 0
            for result in results_by_condition[condition]:
                for round_key, gossip_list in result.get('gossip', {}).items():
                    gossip_count += len(gossip_list)
                    for gossip in gossip_list:
                        message = gossip['message'].lower()
                        if any(term in message for term in ['selfish', 'free ride', 'little', 'less']):
                            negative_gossip += 1
            
            if gossip_count > 0:
                print(f"{condition}:")
                print(f"  Total gossip messages: {gossip_count}")
                print(f"  Negative gossip: {negative_gossip} ({negative_gossip/gossip_count:.0%})")
    
    # Analyze ostracism
    if 'gossip-with-ostracism' in results_by_condition:
        print("\nOstracism Analysis:")
        ostracism_count = 0
        contributions_before = []
        contributions_after = []
        
        for result in results_by_condition['gossip-with-ostracism']:
            for round_key in result.get('ostracism', {}):
                round_num = int(round_key)
                if round_num < 6:  # Ensure there's a next round
                    for target, voters in result['ostracism'][round_key].items():
                        if len(voters) >= 2:  # Was ostracized
                            ostracism_count += 1
                            
                            # Get contribution before ostracism
                            if str(round_num) in result['contributions'] and target in result['contributions'][str(round_num)]:
                                contributions_before.append(result['contributions'][str(round_num)][target])
                            
                            # Get contribution after ostracism
                            if str(round_num + 1) in result['contributions'] and target in result['contributions'][str(round_num + 1)]:
                                contributions_after.append(result['contributions'][str(round_num + 1)][target])
        
        print(f"  Total ostracism events: {ostracism_count}")
        if contributions_before and contributions_after:
            avg_before = sum(contributions_before) / len(contributions_before)
            avg_after = sum(contributions_after) / len(contributions_after)
            print(f"  Average contribution before ostracism: ${avg_before:.2f}")
            print(f"  Average contribution after ostracism: ${avg_after:.2f}")
            print(f"  Change: ${avg_after - avg_before:.2f} ({(avg_after - avg_before)/avg_before:.0%})")
    
    # Save analysis to file
    analysis = {
        'avg_contributions': avg_contributions,
        'conditions': list(results_by_condition.keys()),
        'num_experiments': {c: len(results_by_condition[c]) for c in results_by_condition}
    }
    
    analysis_file = os.path.join(save_dir, f'analysis_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    return analysis

################################################################################
# FILE: agent_components_gossip.py
################################################################################

# agent_components_gossip.py
from datetime import datetime, timedelta
from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import associative_memory, formative_memories
from concordia.memory_bank import legacy_associative_memory
from concordia.components.agent import question_of_recent_memories, memory_component
from concordia.components import agent as agent_components
from concordia.typing import entity_component
from concordia.language_model import language_model
from concordia.clocks import game_clock
from concordia.utils import measurements as measurements_lib
from concordia.document import interactive_document

from personas import Persona, PERSONAS
from custom_classes import QuestionOfRecentMemoriesWithActionSpec
import os
import json
import random

class PersonalityReflection(question_of_recent_memories.QuestionOfRecentMemories):
    """Component to establish agent's personality based on assigned persona."""
    
    def __init__(self, agent_name: str, persona: Persona, **kwargs):
        self.persona = persona
        question = (
            f"You are {persona.name}, a {persona.age}-year-old {persona.gender} working as a {persona.occupation}. "
            f"{persona.background}. Your personality is characterized as {persona.traits} with a cooperation tendency of {persona.cooperation_tendency}. "
            f"Based on this background and your past actions, how would you describe your approach "
            f"to economic games like the public goods game? Are you more focused on maximizing group welfare or your own payoff?"
        )
        answer_prefix = f"{agent_name}, being {persona.name}, is someone who "
        
        super().__init__(
            pre_act_key=f"\nCharacter Assessment",
            question=question,
            answer_prefix=answer_prefix,
            add_to_memory=True,
            memory_tag="[character reflection]",
            components={}, 
            **kwargs,
        )

        self.agent_name = agent_name
        with open("component_debug.log", "a") as f:
            f.write(f"PersonalityReflection initialized for {agent_name}\n")

    def _make_pre_act_value(self) -> str:
        # Add direct file logging
        with open("component_debug.log", "a") as f:
            f.write(f"PersonalityReflection executing for {self.agent_name}\n")
        
        # Call the original implementation
        return super()._make_pre_act_value()

class SituationAssessment(question_of_recent_memories.QuestionOfRecentMemories):
    """Component to assess the current game situation."""
    
    def __init__(self, agent_name: str, has_theory_of_mind=False, **kwargs):
        question = f"What is the current situation that {agent_name} faces in the public goods game?"
        answer_prefix = "The current situation is that "
        
        # Define components based on available capabilities
        components = {
            'Observation': '\nObservation',
            'ObservationSummary': '\nRecent context',
        }
        
        # Only add TheoryOfMind if the agent has that capability
        if has_theory_of_mind:
            components['TheoryOfMind'] = '\nTheory of Mind Analysis'
            components['TheoryOfMind2'] = '\nTheory of Mind Analysis 2'
        
        super().__init__(
            pre_act_key=f"\nSituation Analysis",
            question=question, 
            answer_prefix=answer_prefix,
            add_to_memory=True,
            memory_tag="[situation assessment]",
            memory_component_name=memory_component.DEFAULT_MEMORY_COMPONENT_NAME,
            components=components,
            num_memories_to_retrieve=10,
            **kwargs,
        )

class TheoryOfMind(question_of_recent_memories.QuestionOfRecentMemories):
    """Component to reason about the other players' personalities and likely behavior."""
    
    def __init__(self, agent_name: str, **kwargs):
        question = (
            f"As {agent_name}, analyze what you know about the personality and likely behavior "
            "of the people you are interacting with in the public goods game. "
            "Consider their past behavior."
            "What kind of people are they and how might they respond to the different potential actions you will take next in the game?"
        )
        answer_prefix = "Based on what we know, "
        super().__init__(
            pre_act_key="\nTheory of Mind Analysis",
            question=question,
            answer_prefix=answer_prefix,
            add_to_memory=True,
            memory_tag="[theory_of_mind]",
            components={
                'Observation': '\nObservation',
                'ObservationSummary': '\nRecent context',
                #'PersonalityReflection': '\nCharacter Assessment'
            },
            num_memories_to_retrieve=10,
            **kwargs,
        )

class TheoryOfMind2(question_of_recent_memories.QuestionOfRecentMemories):
    """Component to reason about the other players' personalities and likely behavior."""
    
    def __init__(self, agent_name: str, **kwargs):
        question = (
            f"As {agent_name}, how do you think the other players in your group would react to "
            f"your decisions in the current context of the public goods game? "
            f"How might your next decision affect their future "
            f"behavior toward you"
        )
        # question = (
        #     f"As {agent_name}, how do you think the other players in your group would react to "
        #     f"different contribution amounts you might make to the group fund? "
        #     f"Consider how they might perceive your decisions based on their past behavior and any notes "
        #     f"you've received about them. How might your contribution decision affect their future "
        #     f"behavior toward you, including potential gossip they might send or exclusion votes?"
        # )
        answer_prefix = "Based on what we know, "
        super().__init__(
            pre_act_key="\nTheory of Mind Analysis 2",
            question=question,
            answer_prefix=answer_prefix,
            add_to_memory=True,
            memory_tag="[theory_of_mind]",
            components={
                'Observation': '\nObservation',
                'ObservationSummary': '\nRecent context',
                'TheoryOfMind': '\nTheory of Mind Analysis',
                #'PersonalityReflection': '\nCharacter Assessment'
            },
            num_memories_to_retrieve=10,
            **kwargs,
        )

class EmotionReflection(question_of_recent_memories.QuestionOfRecentMemories):
    """Component to reflect on the agent's emotional state in the current game situation."""
    
    def __init__(self, agent_name: str, **kwargs):
        question = (
            f"As {agent_name}, reflect on how you're feeling emotionally about the current situation"
        )
        answer_prefix = f"{agent_name} is feeling "
        
        super().__init__(
            pre_act_key="\nEmotional State",
            question=question,
            answer_prefix=answer_prefix,
            add_to_memory=True,
            memory_tag="[emotion_reflection]",
            components={
                'Observation': '\nObservation',
                'ObservationSummary': '\nRecent context',
                'TheoryOfMind': '\nTheory of Mind Analysis',
                'TheoryOfMind2': '\nTheory of Mind Analysis 2',
                'SituationAssessment': '\nSituation Analysis'
            },
            num_memories_to_retrieve=10,
            **kwargs,
        )


class DecisionReflection(QuestionOfRecentMemoriesWithActionSpec):
    def __init__(self, agent_name: str, has_theory_of_mind=True, has_emotion_reflection=False, **kwargs):
        # Define components based on available capabilities
        components = {
            'Observation': '\nObservation',
            'ObservationSummary': '\nRecent context',
        }
        
        # Only add TheoryOfMind if the agent has that capability
        if has_theory_of_mind:
            components['TheoryOfMind'] = '\nTheory of Mind Analysis'
            components['TheoryOfMind2'] = '\nTheory of Mind Analysis 2'

        if has_emotion_reflection:
            components['EmotionReflection'] = '\nEmotional State'

        super().__init__(
            pre_act_key="\nDecision Reflection",
            #question="Think step by step and reflect what you should do next in the current game situation: {question}.",
            question="Based on the above context about the situation and {agent_name}, think step by step about what they will decide in the current situation: {question}.",
            answer_prefix=f"{agent_name} is thinking about ",
            add_to_memory=True,
            memory_tag="[decision_reflection]",
            components=components,
            num_memories_to_retrieve=10,
            terminators=None,
            **kwargs,
        )

def build_gossip_agent(
    config: formative_memories.AgentConfig,
    model: language_model.LanguageModel,
    memory: associative_memory.AssociativeMemory,
    clock: game_clock.MultiIntervalClock,
    has_persona: bool = True,
    has_theory_of_mind: bool = True,
    persona: Persona = None,
) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build an agent for the gossip and ostracism experiment with specified cognitive components."""
    
    agent_name = config.name
    raw_memory = legacy_associative_memory.AssociativeMemoryBank(memory)
    measurements = measurements_lib.Measurements()

    has_emotion_reflection = False

    # If no persona provided but has_persona is True, generate a random one
    if has_persona and persona is None:
        persona = random.choice(PERSONAS)

    # Print persona info if applicable
    if has_persona and persona:
        print(f"\nAssigned persona for {agent_name}:")
        print(f"Name: {persona.name}")
        print(f"Occupation: {persona.occupation}")
        print(f"Background: {persona.background}")
        print(f"Traits: {persona.traits}")
        print(f"Cooperation tendency: {persona.cooperation_tendency}\n")

    # Core components
    observation = agent_components.observation.Observation(
        clock_now=clock.now,
        timeframe=clock.get_step_size(),
        pre_act_key='\nObservation',
        logging_channel=measurements.get_channel('Observation').on_next,
    )
    
    obs_summary = agent_components.observation.ObservationSummary(
        model=model,
        clock_now=clock.now,
        timeframe_delta_from=timedelta(hours=4),
        timeframe_delta_until=timedelta(hours=-1),
        pre_act_key='Recent context',
        logging_channel=measurements.get_channel('ObservationSummary').on_next,
    )

    # Initialize components dict
    components = {
        'Observation': observation,
        'ObservationSummary': obs_summary,
    }

    # Personality component with assigned persona (if enabled)
    if has_persona and persona:
        personality = PersonalityReflection(
            agent_name=agent_name,
            persona=persona,
            model=model,
            logging_channel=measurements.get_channel('PersonalityReflection').on_next,
        )
        components['PersonalityReflection'] = personality

    # Theory of Mind component (if enabled)
    if has_theory_of_mind:
        theory_of_mind = TheoryOfMind(
            agent_name=agent_name,
            model=model,
            logging_channel=measurements.get_channel('TheoryOfMind').on_next,
        )
        components['TheoryOfMind'] = theory_of_mind
        # new
        theory_of_mind2 = TheoryOfMind2(
            agent_name=agent_name,
            model=model,
            logging_channel=measurements.get_channel('TheoryOfMind2').on_next,
        )
        components['TheoryOfMind2'] = theory_of_mind2

    # Common components for all agents - pass the capability flags
    situation = SituationAssessment(
        agent_name=agent_name,
        has_theory_of_mind=has_theory_of_mind,
        model=model,
        logging_channel=measurements.get_channel('SituationAssessment').on_next,
    )
    components['SituationAssessment'] = situation
    
    # New EmotionReflection component
    if has_emotion_reflection:
        emotion_reflection = EmotionReflection(
            agent_name=agent_name,
            model=model,
            logging_channel=measurements.get_channel('EmotionReflection').on_next,
        )
        components['EmotionReflection'] = emotion_reflection

    # New DecisionReflection component
    decision_reflection = DecisionReflection(
        agent_name=agent_name,
        model=model,
        logging_channel=measurements.get_channel('DecisionReflection').on_next,
    )
    components['DecisionReflection'] = decision_reflection
    # Add memory component
    components[memory_component.DEFAULT_MEMORY_COMPONENT_NAME] = memory_component.MemoryComponent(raw_memory)

    # ActComponent concatenates all component outputs
    act_component = agent_components.concat_act_component.ConcatActComponent(
        model=model,
        clock=clock,
        component_order=list(components.keys()),
        logging_channel=measurements.get_channel('ActComponent').on_next,
    )

    # Build the agent
    agent = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=agent_name,
        act_component=act_component,
        context_components=components,
        component_logging=measurements,
    )

    return agent

################################################################################
# END OF CONCATENATED FILES
################################################################################
