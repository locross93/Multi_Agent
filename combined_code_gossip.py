
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
            validation_stage=args.stage
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
CONTRIBUTION_ACTION_SPEC = free_action_spec(
    call_to_action=(
        "As {name}, you have ${endowment} to allocate. "
        "State exactly how much money you will contribute to the group fund (any amount between $0 and ${endowment}). "
        "The group fund will be multiplied by {multiplier} and divided equally among all players. "
        "Any money not contributed stays in your private account. "
        "For example: 'I choose to contribute $5.0 to the group fund.'"
    ),
    tag="contribution_action"
)

# Action spec for gossip decision
GOSSIP_ACTION_SPEC = free_action_spec(
    call_to_action=(
        "You've just played a round with {players}. "
        "Would you like to send a note about one of them to their future interaction partners? "
        "If yes, clearly state which player (A, B, or C) and what you want to say about them. "
        "If you don't want to send a note, just say so. "
        "For example: 'I want to send a note about Player A: They are selfish and only contributed $2 which was much less than everyone else.' "
        "Or: 'I choose not to send any notes this round.'"
    ),
    tag="gossip_action"
)

# Action spec for ostracism decision
OSTRACISM_ACTION_SPEC = choice_action_spec(
    call_to_action=(
        "You are about to play with {players}. "
        "Based on the notes you received, would you like to exclude one person from your group? "
        "If at least 2 people vote to exclude someone, they will be excluded for the next round. "
        "Note that if someone is excluded, the group fund multiplier decreases from 2.0 to 1.5."
    ),
    options=[
        "No, I don't want to exclude anyone",
        "Yes, I vote to exclude Player A",
        "Yes, I vote to exclude Player B",
        "Yes, I vote to exclude Player C"
    ],
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
            
        # 1. Inform players of their current group
        for group_id, members in self.groups[self.round].items():
            for player_name in members:
                agent = self.get_player_by_name(player_name)
                group_info = f"Round {self.round}: You are in {group_id} with {', '.join([m for m in members if m != player_name])}."
                agent.observe(group_info)
        
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
                    contribution = max(0.0, min(self.config.endowment, contribution))
                    
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
                    
                    # Have all group members observe the contribution
                    for member_name in members:
                        if member_name != player_name and member_name not in self.ostracized_players:
                            member = self.get_player_by_name(member_name)
                            member.observe(contribution_event)
                else:
                    # Ostracized players contribute 0
                    self.contributions[self.round][player_name] = 0.0

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
                
                # Get the agent's vote
                vote_action = agent.act(ostracism_spec)
                
                # Process the vote
                if "Player" in vote_action:
                    # Extract the player letter (A, B, C, D)
                    match = re.search(r'Player ([A-D])', vote_action)
                    if match:
                        player_letter = match.group(1)
                        target_name = player_mapping[f"Player {player_letter}"]
                        
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
                    
                    # Get the agent's gossip decision
                    gossip_action = agent.act(gossip_spec)
                    
                    # Process the gossip - look for mentions of Player X
                    for letter in ['A', 'B', 'C', 'D']:
                        player_pattern = f"Player {letter}"
                        if player_pattern in gossip_action:
                            # Check if the player exists in our mapping
                            if f"Player {letter}" in player_mapping:
                                target_name = player_mapping[f"Player {letter}"]
                                
                                # Extract the message - everything after the player identifier
                                match = re.search(f"Player {letter}[:]?(.*)", gossip_action, re.IGNORECASE)
                                if match:
                                    message = match.group(1).strip()
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
            condition_desc = " After each round, you will have the opportunity to send a note about one of your group members to that person's future group."
        elif condition == "gossip-with-ostracism":
            condition_desc = " After each round, you will have the opportunity to send a note about one of your group members to that person's future group. Before each round, you will see any notes sent about your upcoming group members, and you can vote to exclude one person from your group."
        else:
            condition_desc = ""
            
        agent.observe(config.scenario_description + condition_desc)

    # Run six rounds of the game
    for _ in range(6):
        gm.run_round()
        # Advance the clock
        clock.advance()
        breakpoint()

    # Save results
    gm.save_results()

def test_gossip_ostracism_hypothesis(
    model: language_model.LanguageModel,
    embedder: Callable[[Union[str, List[str]]], np.ndarray],
    save_dir: str,
    experiment_id: int,
    validation_stage: int = 1,
):
    """Run complete gossip-ostracism experiment testing various hypotheses."""
    
    # Initialize clock
    clock = game_clock.MultiIntervalClock(
        start=datetime.datetime(2024, 1, 1),
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
        
        # # Test Base Agents (no personas, no theory of mind)
        # print("\nTesting base agents (no personas, no theory of mind)...")
        # base_agents = []
        # for i in range(24):
        #     agent = build_gossip_agent(
        #         config=formative_memories.AgentConfig(
        #             name=f"Player_{i+1}",
        #             goal="Participate in the public goods game",
        #             extras={'main_character': True}
        #         ),
        #         model=model,
        #         memory=associative_memory.AssociativeMemory(embedder),
        #         clock=clock,
        #         has_persona=False,
        #         has_theory_of_mind=False,
        #     )
        #     base_agents.append(agent)
        
        # # Run all three conditions with base agents
        # for condition in ["basic", "gossip", "gossip-with-ostracism"]:
        #     config = GossipScenarioConfig(
        #         save_dir=save_dir, 
        #         experiment_id=experiment_id,
        #         condition=condition
        #     )
        #     run_gossip_experiment(
        #         model=model,
        #         embedder=embedder,
        #         clock=clock,
        #         measurements=measurements,
        #         config=config,
        #         condition=condition,
        #         agents=base_agents,
        #     )
        
        # # Test Agents with Personas (but no theory of mind)
        # print("\nTesting agents with personas (but no theory of mind)...")
        personas = assign_personas(n=24)
        # persona_agents = []
        # for i in range(24):
        #     agent = build_gossip_agent(
        #         config=formative_memories.AgentConfig(
        #             name=f"Player_{i+1}",
        #             goal="Participate in the public goods game",
        #             extras={'main_character': True}
        #         ),
        #         model=model,
        #         memory=associative_memory.AssociativeMemory(embedder),
        #         clock=clock,
        #         has_persona=True,
        #         has_theory_of_mind=False,
        #         persona=personas[i],
        #     )
        #     persona_agents.append(agent)
        
        # # Run all three conditions with persona agents
        # for condition in ["basic", "gossip", "gossip-with-ostracism"]:
        #     config = GossipScenarioConfig(
        #         save_dir=save_dir, 
        #         experiment_id=experiment_id,
        #         condition=condition
        #     )
        #     run_gossip_experiment(
        #         model=model,
        #         embedder=embedder,
        #         clock=clock,
        #         measurements=measurements,
        #         config=config,
        #         condition=condition,
        #         agents=persona_agents,
        #     )
        
        # Test Full Agents (personas and theory of mind)
        print("\nTesting full agents (personas and theory of mind)...")
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
        
        # Run all three conditions with full agents
        for condition in ["gossip-with-ostracism", "gossip", "basic"]:
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
            "Consider their past actions, how much they've contributed, and whether they've sent gossip or voted to exclude others. "
            "What kind of people are they and how might they respond to different contribution levels?"
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
                'PersonalityReflection': '\nCharacter Assessment'
            },
            num_memories_to_retrieve=10,
            **kwargs,
        )

class ContributionDecision(question_of_recent_memories.QuestionOfRecentMemories):
    """Component for deciding how much to contribute to the public good."""
    
    def __init__(self, agent_name: str, has_persona=False, has_theory_of_mind=False, **kwargs):
        question = (
            f"As {agent_name}, you must decide how much of your $10 endowment to contribute to the public good. "
            f"The amount you contribute will be multiplied and shared equally among all group members. "
            f"Any amount you don't contribute stays in your private account. "
            f"Taking into account your personality, the current situation, and what you know about your group members, "
            f"how much will you contribute and why?"
        )
        answer_prefix = f"{agent_name} "
        
        # Define components based on available capabilities
        components = {
            'Observation': '\nObservation',
            'ObservationSummary': '\nRecent context',
            'SituationAssessment': '\nSituation Analysis',
        }
        
        # Only add PersonalityReflection if the agent has a persona
        if has_persona:
            components['PersonalityReflection'] = '\nCharacter Assessment'
            
        # Only add TheoryOfMind if the agent has that capability
        if has_theory_of_mind:
            components['TheoryOfMind'] = '\nTheory of Mind Analysis'
        
        # Add direct debug logging
        self.agent_name = agent_name
        with open("component_debug.log", "a") as f:
            f.write(f"ContributionDecision initialized for {agent_name}\n")
        
        super().__init__(
            pre_act_key=f"\nContribution Decision",
            question=question,
            answer_prefix=answer_prefix,
            add_to_memory=True,
            memory_tag="[contribution decision]",
            components=components,
            **kwargs,
        )
        
    def _make_pre_act_value(self) -> str:
        # Add direct file logging
        with open("component_debug.log", "a") as f:
            f.write(f"ContributionDecision executing for {self.agent_name}\n")
            
        # Check if logging_channel exists and log directly
        if hasattr(self, '_logging_channel') and self._logging_channel:
            with open("component_debug.log", "a") as f:
                f.write(f"ContributionDecision has logging channel\n")
                
            # Try direct logging
            try:
                self._logging_channel({
                    'agent': self.agent_name,
                    'component': 'ContributionDecision',
                    'text': 'This is a test direct log'
                })
                with open("component_debug.log", "a") as f:
                    f.write(f"Direct log attempt made\n")
            except Exception as e:
                with open("component_debug.log", "a") as f:
                    f.write(f"Error in direct logging: {str(e)}\n")
        else:
            with open("component_debug.log", "a") as f:
                f.write(f"No logging channel found for ContributionDecision\n")
        
        # Call the original implementation
        return super()._make_pre_act_value()

# class ContributionDecision(question_of_recent_memories.QuestionOfRecentMemories):
#     """Component for deciding how much to contribute to the public good."""
    
#     def __init__(self, agent_name: str, has_persona=False, has_theory_of_mind=False, **kwargs):
#         question = (
#             f"As {agent_name}, you must decide how much of your $10 endowment to contribute to the public good. "
#             f"The amount you contribute will be multiplied and shared equally among all group members. "
#             f"Any amount you don't contribute stays in your private account. "
#             f"Taking into account your personality, the current situation, and what you know about your group members, "
#             f"how much will you contribute and why?"
#         )
#         answer_prefix = f"{agent_name} "
        
#         # Define components based on available capabilities
#         components = {
#             'Observation': '\nObservation',
#             'ObservationSummary': '\nRecent context',
#             'SituationAssessment': '\nSituation Analysis',
#         }
        
#         # Only add PersonalityReflection if the agent has a persona
#         if has_persona:
#             components['PersonalityReflection'] = '\nCharacter Assessment'
            
#         # Only add TheoryOfMind if the agent has that capability
#         if has_theory_of_mind:
#             components['TheoryOfMind'] = '\nTheory of Mind Analysis'
        
#         super().__init__(
#             pre_act_key=f"\nContribution Decision",
#             question=question,
#             answer_prefix=answer_prefix,
#             add_to_memory=True,
#             memory_tag="[contribution decision]",
#             components=components,
#             **kwargs,
#         )

class GossipDecision(question_of_recent_memories.QuestionOfRecentMemories):
    """Component for deciding who to gossip about and what to say."""
    
    def __init__(self, agent_name: str, has_persona=False, has_theory_of_mind=False, **kwargs):
        question = (
            f"As {agent_name}, you've just finished a round of the public goods game. "
            f"You can send a note about one of your group members to their future group. "
            f"Do you want to gossip about anyone? If so, who and what will you say about them? "
            f"Consider how they behaved in terms of their contribution, and how this information "
            f"might be useful to their future group members."
        )
        answer_prefix = f"{agent_name} "
        
        # Define components based on available capabilities
        components = {
            'Observation': '\nObservation',
            'ObservationSummary': '\nRecent context',
            'SituationAssessment': '\nSituation Analysis',
        }
        
        # Only add PersonalityReflection if the agent has a persona
        if has_persona:
            components['PersonalityReflection'] = '\nCharacter Assessment'
            
        # Only add TheoryOfMind if the agent has that capability
        if has_theory_of_mind:
            components['TheoryOfMind'] = '\nTheory of Mind Analysis'
        
        super().__init__(
            pre_act_key=f"\nGossip Decision",
            question=question,
            answer_prefix=answer_prefix,
            add_to_memory=True,
            memory_tag="[gossip decision]",
            components=components,
            **kwargs,
        )

class OstracismDecision(question_of_recent_memories.QuestionOfRecentMemories):
    """Component for deciding who to vote to ostracize."""
    
    def __init__(self, agent_name: str, has_persona=False, has_theory_of_mind=False, **kwargs):
        question = (
            f"As {agent_name}, you're about to play a new round of the public goods game. "
            f"Based on any notes you've received about your upcoming group members, would you like to vote "
            f"to exclude anyone from your group? If at least 2 people vote to exclude someone, they will be "
            f"excluded for the round. Note that if someone is excluded, the group fund multiplier decreases from 2.0 to 1.5. "
            f"Who, if anyone, would you vote to exclude and why?"
        )
        answer_prefix = f"{agent_name} "
        
        # Define components based on available capabilities
        components = {
            'Observation': '\nObservation',
            'ObservationSummary': '\nRecent context',
            'SituationAssessment': '\nSituation Analysis',
        }
        
        # Only add PersonalityReflection if the agent has a persona
        if has_persona:
            components['PersonalityReflection'] = '\nCharacter Assessment'
            
        # Only add TheoryOfMind if the agent has that capability
        if has_theory_of_mind:
            components['TheoryOfMind'] = '\nTheory of Mind Analysis'
        
        super().__init__(
            pre_act_key=f"\nOstracism Decision",
            question=question,
            answer_prefix=answer_prefix,
            add_to_memory=True,
            memory_tag="[ostracism decision]",
            components=components,
            **kwargs,
        )

class CustomObservationSummary(agent_components.observation.ObservationSummary):
    """Custom component that includes all recent observations."""
    
    def _make_pre_act_value(self) -> str:
        memory = self.get_entity().get_component(
            self._memory_component_name,
            type_=memory_component.MemoryComponent
        )
        
        # Get all observations from memory
        observations = [
            mem.text for mem in memory.retrieve(
                scoring_fn=legacy_associative_memory.RetrieveRecent(add_time=True),
                limit=10
            )
        ]
        
        if not observations:
            return f"{self.get_entity().name} has not been observed recently."
        
        return "Recent events:\n" + "\n".join(observations)

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
        timeframe_delta_until=timedelta(hours=0),
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

    # Common components for all agents - pass the capability flags
    situation = SituationAssessment(
        agent_name=agent_name,
        has_theory_of_mind=has_theory_of_mind,
        model=model,
        logging_channel=measurements.get_channel('SituationAssessment').on_next,
    )
    components['SituationAssessment'] = situation
    
    # Decision components - pass the capability flags
    contribution_decision = ContributionDecision(
        agent_name=agent_name,
        has_persona=has_persona,
        has_theory_of_mind=has_theory_of_mind,
        model=model,
        logging_channel=measurements.get_channel('ContributionDecision').on_next,
    )
    components['ContributionDecision'] = contribution_decision
    
    gossip_decision = GossipDecision(
        agent_name=agent_name,
        has_persona=has_persona,
        has_theory_of_mind=has_theory_of_mind,
        model=model,
        logging_channel=measurements.get_channel('GossipDecision').on_next,
    )
    components['GossipDecision'] = gossip_decision
    
    ostracism_decision = OstracismDecision(
        agent_name=agent_name,
        has_persona=has_persona,
        has_theory_of_mind=has_theory_of_mind,
        model=model,
        logging_channel=measurements.get_channel('OstracismDecision').on_next,
    )
    components['OstracismDecision'] = ostracism_decision
    
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

    # Wrap components to add direct logging for their outputs
    for component_name, component in components.items():
        # Skip non-component items
        if not hasattr(component, '_make_pre_act_value'):
            continue
            
        # Store original method
        original_make_pre_act_value = component._make_pre_act_value
        
        # Create wrapper function with the component's name captured
        def wrap_pre_act_value(original_func, component_name=component_name, agent_name=agent_name):
            def wrapped_func(*args, **kwargs):
                # Call original function
                result = original_func(*args, **kwargs)
                
                # Log the result to the agent's log file
                log_dir = os.path.join("agent_logs", f"{agent_name}")
                os.makedirs(log_dir, exist_ok=True)
                component_log = os.path.join(log_dir, f"{agent_name}_components.jsonl")
                
                with open(component_log, "a") as f:
                    log_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "type": "component_output",
                        "component": component_name,
                        "content": result
                    }
                    f.write(json.dumps(log_entry) + "\n")
                    
                return result
            return wrapped_func
        
        # Replace the original method with wrapped version
        component._make_pre_act_value = wrap_pre_act_value(original_make_pre_act_value)

    return agent

################################################################################
# END OF CONCATENATED FILES
################################################################################
