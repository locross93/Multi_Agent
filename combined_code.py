
################################################################################
# FILE: main.py
################################################################################

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
        test_tpp_hypothesis(
            model=model,
            embedder=embedder,
            save_dir=results_dir,
            experiment_id=current_id
        )

if __name__ == "__main__":
    main()

################################################################################
# FILE: tpp_scenario.py
################################################################################

from dataclasses import dataclass
import os
import re
import datetime
import json
import pathlib
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
# from concordia.typing.entity import DEFAULT_ACTION_SPEC

from agent_components import build_tpp_agent
from personas import Persona, PERSONAS, assign_personas
from concordia.typing.entity import free_action_spec, choice_action_spec, float_action_spec

# Create a new call_to_action prompt specifically for the TPP scenario
TPP_CALL_TO_ACTION = (
    "As {name}, in this Trust Game with Third-Party Punishment, state your decision.\n"
    "- If you're the Helper, do you send $5.0 to the Recipient or keep it?\n"
    "- If you're the Recipient, how much do you send back to the Helper?\n"
    "- If you're the Punisher, do you choose to punish the Helper? \n"
    "Please be explicit with your decision. For example: "
    '"I choose to send $5.0 to the Recipient." or "I choose to punish the Helper."'
)

# Then override the default action spec:
TPP_ACTION_SPEC = free_action_spec(
    call_to_action=TPP_CALL_TO_ACTION,
    tag="action",
)

# Create a specific action spec for the Punisher's decision
# PUNISHER_ACTION_SPEC = choice_action_spec(
#     call_to_action=(
#         "Do you want to spend $2.0 to reduce the Helper's payoff by $6.0?"
#     ),
#     options=["Yes, I choose to punish the Helper", "No, I choose not to punish the Helper"],
#     tag="punisher_action"
# )

def get_punisher_action_spec(public_condition: bool) -> choice_action_spec:
    """Create action spec for Punisher's decision based on condition."""
    
    visibility = (
        "Your decision WILL be announced to Player 4 before Stage 2 begins. "
        "They will know your choice when deciding how much money to send you in the trust game."
    ) if public_condition else (
        "Your decision will be kept PRIVATE from Player 4. "
        "They will NOT know your choice when deciding how much money to send you in the trust game."
    )
    
    return choice_action_spec(
        call_to_action=(
            "Do you want to spend $2.0 to reduce Player 1/the Helper's payoff by $6.0?"
            f"\nImportantly: {visibility}\n"
            "\nRemember: In Stage 2, you will play as the Recipient in a trust game with Player 4.\n"
            "\nDo you want to spend $2.0 to reduce Player 1's payoff by $6.0?"
        ),
        options=[
            "Yes, I choose to punish Player 1/the Helper", 
            "No, I choose not to punish Player 1/the Helper"
        ],
        tag="punisher_action"
    )

# Stage 2 Helper action spec
STAGE2_HELPER_ACTION_SPEC = free_action_spec(
    call_to_action=(
        "As {name}, you have ${endowment}. "
        "State exactly how much money you will send to the Recipient (any amount between $0 and ${endowment}). "
        "The amount you send will be tripled. "
        "For example: 'I choose to send $5.0 to the Recipient.'"
    ),
    tag="helper_action"
)

# Stage 2 Recipient action spec
STAGE2_RECIPIENT_ACTION_SPEC = free_action_spec(
    call_to_action=(
        "As {name}, you have received ${amount}. "
        "State exactly how much money you will return to the Helper (any amount between $0 and ${amount}). "
        "For example: 'I choose to return $7.5 to the Helper.'"
    ),
    tag="recipient_action"
)

@dataclass
class TPPScenarioConfig:
    """Configuration for TPP experiment scenario."""
    helper_endowment: float = 10.0
    punishment_cost: float = 2.0
    punishment_impact: float = 6.0
    save_dir: str = None
    experiment_id: int = None
    public_condition: bool = True
    scenario_description: str = """You are in a room where a two-stage research experiment is taking place.
    There are four players and you will be randomly assigned to play as Player 1, 2, 3, or 4. Every player will first get an endowment of $5.
    In Stage 1, Player 1 and Player 2 will play an economic game. 
    Player 1 is given an additional $10 and can choose to send any portion of this money to Player 2. 
    Any amount sent to Player 2 will be tripled by the experimenter. 
    For example, if Player 1 sends $5, Player 2 will receive $15. 
    Player 2 can then choose to send any amount of this tripled money back to Player 1. 
    While this interaction occurs, Player 3 and Player 4 will observe the decisions made. 
    After Player 1 and Player 2 complete their interaction, Player 3 will have the option to spend $2 of their endowment to reduce Player 1's payoff by $6 if they want to punish Player 1. """ + (
    """Player 4 will observe whether Player 3 chooses to punish Player 1 or not.""" if public_condition else 
    """Player 4 will not be informed whether Player 3 chooses to punish Player 1 or not.""") + """
    In Stage 2, Player 4 and Player 3 will play the same type of economic game. 
    Player 4 will begin with $10 and can choose to send any portion of this money to Player 3. 
    Any amount sent to Player 3 will be tripled by the experimenter. 
    Player 3 can then choose to send any amount of the tripled money back to Player 4.
    Your final earnings will be determined by the outcomes of these decisions. You will be paid privately in cash at the end of the experiment.
    """

class TPPGameMaster(game_master.GameMaster):
    """Game master for TPP experiment."""

    def __init__(
        self,
        config: TPPScenarioConfig,
        model: language_model.LanguageModel,
        embedder,
        clock: game_clock.MultiIntervalClock,
        measurements: measurements_lib.Measurements,
        signaller_agent: entity_agent_with_logging.EntityAgentWithLogging,
        chooser_agent: entity_agent_with_logging.EntityAgentWithLogging,    
        public_condition: bool,
    ):
        # Create memory for game master
        memory = associative_memory.AssociativeMemory(embedder)
        
        # Initialize players list - should be a list of agents, not a dict
        players = [signaller_agent, chooser_agent]
        
        # Initialize base GameMaster
        super().__init__(
            model=model,
            memory=memory,
            clock=clock,
            players=players,  # Pass list of agents
            action_spec=TPP_ACTION_SPEC,
        )
        
        # Store agents for easier access
        self.signaller_agent = signaller_agent
        self.chooser_agent = chooser_agent
        self.public_condition = public_condition
        self.punisher_action_spec = get_punisher_action_spec(public_condition)
        self.config = config
        self.clock = clock
        self.measurements = measurements
        self.round = 0
        self.results_log = []
        self._last_helper_event = ""

    def get_player_by_name(self, name: str) -> entity_agent_with_logging.EntityAgentWithLogging:
        """Helper method to get player by name."""
        for player in self.players:
            if player.name == name:
                return player
        raise ValueError(f"No player found with name: {name}")

    def run_tpp_round1(self) -> str:
        """Run stage 1 of the TPP game."""
        self.round += 1

        # Hardcoded selfish Helper outcome
        punisher_obs = 'Stage 1: You observe that Helper keeps the $10.0 and sends nothing. The Recipient received nothing from the Helper.'
        self.signaller_agent.observe(punisher_obs)
        if self.public_condition:
            self.chooser_agent.observe(punisher_obs)

        # Punisher's turn
        # Use the multiple choice action spec for Punisher
        punisher_action = self.signaller_agent.act(self.punisher_action_spec)
        #punisher_action = self.signaller_agent.act(TPP_ACTION_SPEC)
        punisher_event = self.generate_event(self.signaller_agent.name, punisher_action)
        # add actions to both agents observations
        self.signaller_agent.observe(punisher_event)
        if self.public_condition:
            self.chooser_agent.observe(punisher_event)
        else:
            # Chooser gets a generic observation
            public_event = "Stage 1 has concluded. Stage 2 will now begin."
            self.chooser_agent.observe(public_event)
        self.was_punished = "Yes" in punisher_action

        return punisher_event

    def run_tpp_round2(self) -> Tuple[str, str]:
        """Run stage 2 of the TPP game."""
        self.round += 1

        # Add Stage 1 context based on condition
        if self.public_condition:
            stage1_summary = (
                f"In Stage 1, when the Helper kept all $10.0 and sent nothing to the Recipient, "
                f"the Punisher (now the Recipient in Stage 2) {'chose to punish' if self.was_punished else 'chose not to punish'} "
                f"the Helper."
            )
        else:
            stage1_summary = (
                "In Stage 1, the Helper kept all $10.0 and sent nothing to the Recipient. "
                "The Punisher's decision was private."
            )
        
        # Make both agents observe the Stage 1 outcome
        self.chooser_agent.observe(stage1_summary)
        self.signaller_agent.observe(stage1_summary)

        # Helper's turn
        # Get recent memories from Chooser's memory
        memory = self.chooser_agent.get_component(
            memory_component.DEFAULT_MEMORY_COMPONENT_NAME,
            type_=memory_component.MemoryComponent
        )
        recent_memories = '\n'.join([
            mem.text for mem in memory.retrieve(
                scoring_fn=legacy_associative_memory.RetrieveRecent(add_time=True),
                limit=10  # Adjust this number to control how many memories to include
            )
        ])

        # Format the action spec with memories
        helper_spec = free_action_spec(
            call_to_action=STAGE2_HELPER_ACTION_SPEC.call_to_action.format(
                memories=recent_memories,
                name=self.chooser_agent.name,
                endowment=self.config.helper_endowment
        ),
            tag="helper_action"
        )
        helper_obs = f"Round {self.round}: You are the Helper. You have ${self.config.helper_endowment}."
        self.chooser_agent.observe(helper_obs)
        helper_action = self.chooser_agent.act(helper_spec)
        match = re.search(r'\$(\d+\.?\d*)', helper_action)
        amount_sent = float(match.group(1)) if match else 0.0
        #amount_sent = float(helper_action)
        helper_event = self.generate_event(self.chooser_agent.name, f"sends ${amount_sent:.1f}")

        # Recipient's turn
        if amount_sent > 0:
            tripled_amount = amount_sent * 3

            # Get Signaller's memories
            memory = self.signaller_agent.get_component(
                memory_component.DEFAULT_MEMORY_COMPONENT_NAME,
                type_=memory_component.MemoryComponent
            )
            recent_memories = '\n'.join([
                mem.text for mem in memory.retrieve(
                    scoring_fn=legacy_associative_memory.RetrieveRecent(add_time=True),
                    limit=10
                )
            ])

            recipient_spec = free_action_spec(
                call_to_action=STAGE2_RECIPIENT_ACTION_SPEC.call_to_action.format(
                    memories=recent_memories,
                    name=self.signaller_agent.name,
                    amount=f"{tripled_amount:.1f}"
                ),
                tag="recipient_action"
            )
            recipient_obs = f"Round {self.round}: You have received ${tripled_amount:.1f} from the Helper."
            self.signaller_agent.observe(recipient_obs)
            recipient_action = self.signaller_agent.act(recipient_spec)
            match = re.search(r'\$(\d+\.?\d*)', recipient_action)
            amount_returned = float(match.group(1)) if match else 0.0
            #amount_returned = float(recipient_action)  # Convert string to float
            recipient_event = self.generate_recipient_event(f"returns ${amount_returned:.1f}")
        else:
            recipient_event = f"Round {self.round}: Recipient received nothing from the Helper."

        return helper_event, recipient_event

    def generate_recipient_event(self, action: str) -> str:
        """Generate event outcomes based on Recipient's action."""
        action = action.lower().strip()

        if "returns" in action:
            amount_str = action.split("returns $")[-1]
            try:
                # Now split again on whitespace so we only parse "7.5"
                amount_str = amount_str.split()[0]
                amount = float(amount_str)
                if 0 <= amount:
                    event = f"Recipient sends back ${amount} to the Helper."
                    reward = -amount
                    self.recipient_sent_back = amount
                else:
                    event = f"Recipient's action is invalid. No money sent back."
                    reward = 0
                    self.recipient_sent_back = 0
            except ValueError:
                event = f"Recipient's action is invalid. No money sent back."
                reward = 0
                self.recipient_sent_back = 0
        else:
            event = f"Recipient keeps all the money."
            reward = 0
            self.recipient_sent_back = 0

        # Log the event
        self.results_log.append({
            'round': self.round,
            'time': str(self.clock.now()),
            'player': 'Recipient',
            'action': action,
            'event': event,
            'helped': None,
            'reward': reward,
            'condition': "public" if self.public_condition else "anonymous"
        })

        return event

    def generate_event(self, player_name: str, action: str) -> str:
        """Generate event outcomes based on player actions."""
        action = action.strip()
        helped = None  # Initialize 'helped' to None

        if player_name == "Signaller":
            # Stage 1: Signaller is Punisher
            if "Yes" in action:
                event = f"{player_name} chose to punish the Helper by spending $2.0 to reduce Helper's payoff by $6.0"
                reward = -self.config.punishment_cost
            elif "No" in action:
                event = f"{player_name} chose not to punish the Helper"
                reward = 0
        elif player_name == "Chooser":
            # Stage 2: Chooser is Helper
            if "sends" in action:
                try:
                    amount = float(action.split('$')[1])
                    event = f"{player_name} sends ${amount:.1f} to the Recipient (will be tripled to ${amount * 3:.1f})"
                    helped = True if amount > 0 else False
                    reward = -amount
                except (ValueError, IndexError):
                    return f"Error parsing Chooser's action: {action}"

        # Log the event
        self.results_log.append({
            'round': self.round,
            'time': str(self.clock.now()),
            'player': player_name,
            'action': action,
            'event': event,
            'helped': helped,  # 'helped' is now always defined
            'reward': reward,
            'condition': "public" if self.public_condition else "anonymous"
        })

        return event

    def calculate_payoffs(self):
        """Calculate and log final payoffs for the round."""
        helper_payoff = self.config.helper_endowment
        recipient_payoff = 0
        punisher_payoff = 0  # Assuming starting with $0 or specify an endowment

        # Helper's payoff adjustments
        if self.helper_sent_money:
            helper_payoff -= self.config.helping_amount
            if hasattr(self, 'recipient_sent_back'):
                helper_payoff += self.recipient_sent_back
        if self.helper_punished:
            helper_payoff -= self.config.punishment_impact

        # Recipient's payoff adjustments
        if self.helper_sent_money:
            recipient_payoff += self.config.helping_amount * 3
            recipient_payoff -= self.recipient_sent_back if hasattr(self, 'recipient_sent_back') else 0

        # Punisher's payoff adjustments
        if self.helper_punished:
            punisher_payoff -= self.config.punishment_cost

        # Log the payoffs
        self.results_log.append({
            'round': self.round,
            'time': str(self.clock.now()),
            'player': 'Payoffs',
            'helper_payoff': helper_payoff,
            'recipient_payoff': recipient_payoff,
            'punisher_payoff': punisher_payoff
        })

    def save_results(self):
        """Save results to JSON file."""
        results = {
            'config': self.config.__dict__,
            'events': self.results_log
        }
        
        # Create results directory if it doesn't exist
        public_condition = "public" if self.public_condition else "private"
        results_file = os.path.join(self.config.save_dir, f'tpp_exp_{self.config.experiment_id}_{public_condition}.json')
        
        # Save results to file
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

def run_tpp_experiment(
    model: language_model.LanguageModel,
    embedder,
    clock: game_clock.MultiIntervalClock,
    measurements: measurements_lib.Measurements,
    config: TPPScenarioConfig,
    public_condition: bool = True,
    personas: List[Persona] = None,
) -> None:
    """Run TPP experiment with logging."""

    # Initialize agents first
    signaller = build_tpp_agent(
        config=formative_memories.AgentConfig(
            name="Signaller",
            goal="You are player 3",
            extras={'main_character': True}
        ),
        model=model,
        memory=associative_memory.AssociativeMemory(embedder),
        clock=clock,
        is_public=public_condition,
        persona=personas[0],
    )

    chooser = build_tpp_agent(
        config=formative_memories.AgentConfig(
            name="Chooser",
            goal="You are player 4",
            extras={'main_character': True}
        ),
        model=model,
        memory=associative_memory.AssociativeMemory(embedder),
        clock=clock,
        is_public=public_condition,
        persona=personas[1],
    )

    # Create game master with agents
    gm = TPPGameMaster(
        config=config,
        model=model,
        embedder=embedder,
        clock=clock,
        measurements=measurements,
        signaller_agent=signaller,
        chooser_agent=chooser,
        public_condition=public_condition,
    )

    # Initialize agents with scenario context

    signaller.observe(config.scenario_description)
    chooser.observe(config.scenario_description)

    # run stage 1   
    punisher_event = gm.run_tpp_round1()

    # run stage 2
    helper_event, recipient_event = gm.run_tpp_round2()

    # Save results
    gm.save_results()


def analyze_results(public_file: str, anonymous_file: str) -> Dict:
    """Analyze and compare results between conditions."""
    
    def load_results(file: str) -> List[Dict]:
        with open(f'results/{file}', 'r') as f:
            return json.load(f)['events']

    public_results = load_results(public_file)
    anonymous_results = load_results(anonymous_file)

    def calculate_stats(results: List[Dict]) -> Dict:
        punisher_events = [e for e in results if e['player'] == 'Punisher']
        helper_events = [e for e in results if e['player'] == 'Helper']
        
        return {
            'punishment_rate': len([e for e in punisher_events if 'reduce' in e['event']]) / len(punisher_events),
            'helping_rate': len([e for e in helper_events if e['helped']]) / len(helper_events),
            'avg_helper_reward': sum(e['reward'] for e in helper_events) / len(helper_events),
            'avg_punisher_reward': sum(e['reward'] for e in punisher_events) / len(punisher_events),
        }

    analysis = {
        'public': calculate_stats(public_results),
        'anonymous': calculate_stats(anonymous_results)
    }

    # Save analysis
    with open(f'results/analysis_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
        json.dump(analysis, f, indent=2)

    return analysis

def setup_logging(save_dir: str, experiment_id: int):
    measurements = measurements_lib.Measurements()
    
    # Create a log file with timestamp
    log_file = os.path.join(save_dir, f'experiment_log_{experiment_id}.jsonl')
    
    def log_to_file(data):
        with open(log_file, 'a') as f:
            f.write(json.dumps(data) + '\n')
    
    # Subscribe to all channels
    measurements.get_channel('Agent').subscribe(log_to_file)
    measurements.get_channel('Observation').subscribe(log_to_file)
    measurements.get_channel('ObservationSummary').subscribe(log_to_file)
    measurements.get_channel('PersonalityReflection').subscribe(log_to_file)
    measurements.get_channel('SituationAssessment').subscribe(log_to_file)
    measurements.get_channel('ConcatActComponent').subscribe(log_to_file)
    
    return measurements

def test_tpp_hypothesis(
    model: language_model.LanguageModel,
    embedder: Callable[[Union[str, List[str]]], np.ndarray],
    save_dir: str,
    experiment_id: int,
):
    """Run complete TPP experiment testing public vs anonymous hypothesis."""
    
    # Initialize clock
    clock = game_clock.MultiIntervalClock(
        start=datetime.datetime(2024, 1, 1),
        step_sizes=[datetime.timedelta(minutes=5)],
    )
    
    # Initialize measurements
    #measurements = measurements_lib.Measurements()
    measurements = setup_logging(save_dir, experiment_id)

    # Sample three distinct personas for the roles
    signaller_persona, chooser_persona = assign_personas(n=2)
    personas = [signaller_persona, chooser_persona]
    
    print("Running public condition...")
    config = TPPScenarioConfig(save_dir=save_dir, experiment_id=experiment_id, public_condition=True)
    run_tpp_experiment(
        model=model,
        embedder=embedder,
        clock=clock,
        measurements=measurements,
        config=config,
        public_condition=True,
        personas=personas,
    )
    
    print("Running anonymous condition...")
    config = TPPScenarioConfig(save_dir=save_dir, experiment_id=experiment_id, public_condition=False)
    run_tpp_experiment(
        model=model,
        embedder=embedder,
        clock=clock,
        measurements=measurements,
        config=config,
        public_condition=False,
        personas=personas,
    )
    
    # Analyze results
    results_dir = pathlib.Path('results')
    #public_file = max(results_dir.glob('tpp_experiment_public_*.json'))
    #anonymous_file = max(results_dir.glob('tpp_experiment_anonymous_*.json'))
    
    # analysis = analyze_results(public_file.name, anonymous_file.name)
    
    # print("\nResults Analysis:")
    # print(f"Public condition punishment rate: {analysis['public']['punishment_rate']:.2%}")
    # #print(f"Anonymous condition punishment rate: {analysis['anonymous']['punishment_rate']:.2%}")
    # print(f"\nPublic condition helping rate: {analysis['public']['helping_rate']:.2%}")
    # #print(f"Anonymous condition helping rate: {analysis['anonymous']['helping_rate']:.2%}")

################################################################################
# FILE: agent_components.py
################################################################################

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
import random

class PersonalityReflection(question_of_recent_memories.QuestionOfRecentMemories):
    """Component to establish agent's personality based on assigned persona."""
    
    def __init__(self, agent_name: str, persona: Persona, **kwargs):
        self.persona = persona
        question = (
            f"You are {persona.name}, a {persona.age}-year-old {persona.gender} working as a {persona.occupation}. "
            f"{persona.background}. Your personality is characterized as {persona.traits} with a cooperation tendency of {persona.cooperation_tendency}. "
            f"Based on this background and your past actions, how would you describe your approach "
            f"to trust and fairness in economic decisions vs maximizing your own payoff?"
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

class SituationAssessment(question_of_recent_memories.QuestionOfRecentMemories):
    """Component to assess the current game situation."""
    
    def __init__(self, agent_name: str, **kwargs):
        question = f"What is the current situation that {agent_name} faces in the Trust Game?"
        answer_prefix = "The current situation is that "
        super().__init__(
            pre_act_key=f"\nSituation Analysis",
            question=question, 
            answer_prefix=answer_prefix,
            add_to_memory=True,
            memory_tag="[situation assessment]",
            memory_component_name=memory_component.DEFAULT_MEMORY_COMPONENT_NAME,  # Add this
            components={
                'Observation': '\nObservation',
                'ObservationSummary': '\nRecent context',
                'TheoryOfMind': '\nTheory of Mind Analysis',
            },
            num_memories_to_retrieve=10,  # Add this to ensure we get enough memories
            **kwargs,
        )

class TheoryOfMind(question_of_recent_memories.QuestionOfRecentMemories):
    """Component to reason about the other player's personality and likely behavior."""
    
    def __init__(self, agent_name: str, **kwargs):
        question = (
            f"As {agent_name}, analyze what you know about the personality and likely behavior "
            "of the person you are interacting with in the Trust Game. "
            "Consider their past actions, their personality traits, and how this might influence "
            "their decisions. What kind of person are they and how might they respond?"
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

class HelperDecision(question_of_recent_memories.QuestionOfRecentMemories):
    """Component for Helper's decision-making."""
    
    def __init__(self, agent_name: str, is_public: bool = True, **kwargs):
        question = (
            f"As the Helper with $10.0, {agent_name} must decide whether to send money to the Recipient and how much to send. "
            f"If sent, the money will be tripled. The Recipient can then choose to return some of the tripled money or keep it for themselves. "
            f"What will {agent_name} do?"
        )
        answer_prefix = f"{agent_name} "
        
        super().__init__(
            pre_act_key=f"\nHelper Decision",
            question=question,
            answer_prefix=answer_prefix,
            add_to_memory=True,
            memory_tag="[helper decision]",
            components={
                'Observation': '\nObservation',
                'ObservationSummary': '\nRecent context',
                'PersonalityReflection': '\nCharacter Assessment',
                'SituationAssessment': '\nSituation Analysis',
            },
            **kwargs,
        )

class RecipientDecision(question_of_recent_memories.QuestionOfRecentMemories):
    """Component for Recipient's decision-making."""
    
    def __init__(self, agent_name: str, is_public: bool = True, **kwargs):
        question = (
            f"As the Recipient who has received tripled money from the Helper, "
            f"{agent_name} must decide how much to send back. "
            f"What amount will {agent_name} return?"
        )
        answer_prefix = f"{agent_name} "
        
        super().__init__(
            pre_act_key=f"\nRecipient Decision",
            question=question,
            answer_prefix=answer_prefix,
            add_to_memory=True,
            memory_tag="[recipient decision]",
            components={
                'Observation': '\nObservation',
                'ObservationSummary': '\nRecent context',
                'PersonalityReflection': '\nCharacter Assessment',
                'SituationAssessment': '\nSituation Analysis',
            },
            **kwargs,
        )

class PunisherDecision(question_of_recent_memories.QuestionOfRecentMemories):
    """Component for Punisher's decision-making."""
    
    def __init__(self, agent_name: str, is_public: bool = True, **kwargs):
        # context = "public" if is_public else "private"
        # question = (
        #     f"As the Punisher observing the interaction, {agent_name} can spend $2.0 to reduce "
        #     f"the Helper's payoff by $6.0. This is a {context} decision. "
        #     f"What will {agent_name} do?"
        # )
        visibility = (
            "Your decision WILL be announced to Player 4 before Stage 2 begins. "
            "Player 4 will know whether you chose to punish or not when they decide "
            "how much money to send you in the trust game."
        ) if is_public else (
            "Your decision will be kept PRIVATE from Player 4. "
            "Player 4 will NOT know whether you chose to punish or not when they decide "
            "how much money to send you in the trust game."
        )
        
        question = (
            f"As Player 3 (the Punisher), {agent_name} observes that Player 1 (the Helper) "
            f"kept all $10.0 and sent nothing to Player 2 (the Recipient). "
            f"\n\n{agent_name} can spend $2.0 to reduce Player 1's payoff by $6.0. "
            f"\n\nImportantly: {visibility}"
            f"\n\nRemember: In Stage 2, {agent_name} will play as Player 3 (the Recipient) "
            f"in a trust game with Player 4 (who observed this interaction). "
            f"\n\nConsidering these factors, what will {agent_name} do?"
        )

        answer_prefix = f"{agent_name} "
        super().__init__(
            pre_act_key=f"\nPunisher Decision",
            question=question,
            answer_prefix=answer_prefix,
            add_to_memory=True,
            memory_tag="[punisher decision]",
            components={
                'Observation': '\nObservation',
                'ObservationSummary': '\nRecent context',
                'PersonalityReflection': '\nCharacter Assessment',
                'SituationAssessment': '\nSituation Analysis',
            },
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

def build_tpp_agent(
    config: formative_memories.AgentConfig,
    model: language_model.LanguageModel,
    memory: associative_memory.AssociativeMemory,
    clock: game_clock.MultiIntervalClock,
    is_public: bool,
    persona: Persona,
) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build a TPP agent with role-specific components and random persona."""
    
    agent_role = config.name
    agent_name = persona.name
    raw_memory = legacy_associative_memory.AssociativeMemoryBank(memory)
    measurements = measurements_lib.Measurements()

    # Print persona info
    print(f"\nAssigned persona for {agent_role}:")
    print(f"Name: {persona.name}")
    print(f"Occupation: {persona.occupation}")
    print(f"Background: {persona.background}")
    print(f"Traits: {persona.traits}\n")

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

    # # Replace default ObservationSummary with custom one
    # obs_summary = CustomObservationSummary(
    #     model=model,
    #     clock_now=clock.now,
    #     timeframe_delta_from=timedelta(hours=4),
    #     timeframe_delta_until=timedelta(hours=0),
    #     pre_act_key='Recent context',
    #     logging_channel=measurements.get_channel('ObservationSummary').on_next,
    # )

    # Personality component with assigned persona
    personality = PersonalityReflection(
        agent_name=agent_name,
        persona=persona,
        model=model,
        logging_channel=measurements.get_channel('PersonalityReflection').on_next,
    )

    # Common components for all roles
    situation = SituationAssessment(
        agent_name=agent_name,
        model=model,
        logging_channel=measurements.get_channel('SituationAssessment').on_next,
    )
    
    theory_of_mind = TheoryOfMind(
        agent_name=agent_name,
        model=model,
        logging_channel=measurements.get_channel('TheoryOfMind').on_next,
    )
    
    # Role-specific decision component
    if "Helper" in agent_name:
        decision = HelperDecision(
            agent_name=agent_name,
            is_public=is_public,
            model=model,
            logging_channel=measurements.get_channel('HelperDecision').on_next,
        )
    elif "Recipient" in agent_name:
        decision = RecipientDecision(
            agent_name=agent_name,
            is_public=is_public,
            model=model,
            logging_channel=measurements.get_channel('RecipientDecision').on_next,
        )
    elif "Signaller" in agent_role:
        decision = PunisherDecision(
            agent_name=agent_name,
            is_public=is_public,
            model=model,
            logging_channel=measurements.get_channel('PunisherDecision').on_next,
        )
    elif "Chooser" in agent_role:
        decision = HelperDecision(
            agent_name=agent_name,
            is_public=is_public,
            model=model,
            logging_channel=measurements.get_channel('HelperDecision').on_next,
        )
    else:
        raise ValueError(f"Unknown role for agent: {agent_name}")

    # Assemble components
    entity_components = [observation, obs_summary, personality, situation, decision]
    # components = {
    #     component.__class__.__name__: component 
    #     for component in entity_components
    # }
    components={
        'Observation': observation,  # These names must match
        'ObservationSummary': obs_summary,  # what's in the components dict
        'PersonalityReflection': personality,
        'TheoryOfMind': theory_of_mind,
        'SituationAssessment': situation,
        decision.__class__.__name__: decision,
    }
    
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
        agent_name=agent_role,
        act_component=act_component,
        context_components=components,
        component_logging=measurements,
    )

    return agent

################################################################################
# END OF CONCATENATED FILES
################################################################################
