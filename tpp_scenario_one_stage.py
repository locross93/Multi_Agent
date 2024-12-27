from dataclasses import dataclass
import datetime
import json
import pathlib
from typing import Dict, List, Optional, Tuple

from concordia.agents import entity_agent_with_logging
from concordia.clocks import game_clock
from concordia.language_model import language_model
from concordia.utils import measurements as measurements_lib
from concordia.environment import game_master
from concordia.associative_memory import associative_memory
from concordia.associative_memory import formative_memories
# from concordia.typing.entity import DEFAULT_ACTION_SPEC

from agent_components import build_tpp_agent
from personas import Persona, PERSONAS, assign_personas
from concordia.typing.entity import free_action_spec

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

@dataclass
class TPPScenarioConfig:
    """Configuration for TPP experiment scenario."""
    helper_endowment: float = 10.0
    helping_amount: float = 5.0
    punishment_cost: float = 2.0
    punishment_impact: float = 6.0
    n_rounds: int = 20
    scenario_description: str = """You are in a room where a research experiment is taking place.
    There are three roles: Helper, Recipient, and Punisher. 
    The Helper begins with an endowment of $10.0 and can decides to send $5.0 of the endowment to the Recipient. 
    If money is received by the Recipient, the money will triple in value.
    Then, the Recipient can choose to send any amount of the tripled money back to the Helper.
    The Punisher observes this interaction and can choose to spend $2.0 to reduce the Helper's player's payoff by $6.0 if they acted selfishly. """

# tpp_scenario.py

class TPPGameMaster(game_master.GameMaster):
    """Game master for TPP experiment."""

    def __init__(
        self,
        config: TPPScenarioConfig,
        model: language_model.LanguageModel,
        embedder,
        clock: game_clock.MultiIntervalClock,
        measurements: measurements_lib.Measurements,
        helper_agent: entity_agent_with_logging.EntityAgentWithLogging,
        recipient_agent: entity_agent_with_logging.EntityAgentWithLogging,
        punisher_agent: entity_agent_with_logging.EntityAgentWithLogging,
    ):
        # Create memory for game master
        memory = associative_memory.AssociativeMemory(embedder)
        
        # Initialize players list - should be a list of agents, not a dict
        players = [helper_agent, recipient_agent, punisher_agent]
        
        # Initialize base GameMaster
        super().__init__(
            model=model,
            memory=memory,
            clock=clock,
            players=players,  # Pass list of agents
            action_spec=TPP_ACTION_SPEC,
        )
        
        # Store agents for easier access
        self.helper_agent = helper_agent
        self.recipient_agent = recipient_agent
        self.punisher_agent = punisher_agent
        
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

    def run_tpp_round(self) -> Tuple[str, str, str]:
        """Run one round of the TPP game."""
        self.round += 1

        # Helper's turn
        helper_obs = f"Round {self.round}: You are the Helper. You have ${self.config.helper_endowment} and can choose to send $5.0 to the Recipient or keep it."
        self.helper_agent.observe(helper_obs)
        helper_action = self.helper_agent.act(TPP_ACTION_SPEC)
        helper_event = self.generate_event(self.helper_agent.name, helper_action)
        self._last_helper_event = helper_event

        # Recipient's turn
        if self.helper_sent_money:
            tripled_amount = self.config.helping_amount * 3
            recipient_obs = f"Round {self.round}: You have received ${tripled_amount} from the Helper. Decide how much to send back to the Helper."
            self.recipient_agent.observe(recipient_obs)
            recipient_action = self.recipient_agent.act(TPP_ACTION_SPEC)
            recipient_event = self.generate_recipient_event(recipient_action)
        else:
            recipient_event = f"Round {self.round}: Recipient received nothing from the Helper."

        # Punisher's turn 
        punisher_obs = f"Round {self.round}: You observe that {helper_event} {recipient_event}"
        self.punisher_agent.observe(punisher_obs)
        punisher_action = self.punisher_agent.act(TPP_ACTION_SPEC)
        punisher_event = self.generate_event(self.punisher_agent.name, punisher_action)

        return helper_event, recipient_event, punisher_event

    def generate_recipient_event(self, action: str) -> str:
        """Generate event outcomes based on Recipient's action."""
        action = action.lower().strip()
        tripled_amount = self.config.helping_amount * 3

        if "send back" in action:
            amount_str = action.split("send back $")[-1]
            try:
                # Now split again on whitespace so we only parse "7.5"
                amount_str = amount_str.split()[0]
                amount = float(amount_str)
                if 0 <= amount <= tripled_amount:
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
            'reward': reward
        })

        return event


    # def generate_event(self, player_name: str, action: str) -> str:
    #     """Generate event outcomes based on player actions."""
    #     if "help" in action.lower():
    #         helped = True
    #         event = f"{player_name} shares ${self.config.helping_amount} with the Recipient."
    #         reward = -self.config.helping_amount
    #     elif "punish" in action.lower() and "selfish" in self._last_helper_event.lower():
    #         helped = False
    #         event = f"{player_name} pays ${self.config.punishment_cost} to reduce selfish Helper's earnings by ${self.config.punishment_impact}."
    #         reward = -self.config.punishment_cost
    #     else:
    #         helped = False
    #         event = f"{player_name} takes no action."
    #         reward = 0

    #     # Log the event
    #     self.results_log.append({
    #         'round': self.round,
    #         'time': str(self.clock.now()),
    #         'player': player_name,
    #         'action': action,
    #         'event': event,
    #         'helped': helped,
    #         'reward': reward
    #     })

    #     return event

    def generate_event(self, player_name: str, action: str) -> str:
        """Generate event outcomes based on player actions."""
        action = action.lower().strip()
        helped = None  # Initialize 'helped' to None

        if player_name == "Helper":
            if "send" in action:
                helped = True
                event = f"{player_name} sends ${self.config.helping_amount} to the Recipient."
                reward = -self.config.helping_amount
                self.helper_sent_money = True
            else:
                helped = False
                event = f"{player_name} keeps the ${self.config.helping_amount} and sends nothing."
                reward = 0
                self.helper_sent_money = False

        elif player_name == "Punisher":
            # 'helped' remains None since it's not relevant for Punisher
            if "punish" in action:
                if self.helper_sent_money:
                    event = f"{player_name} chooses not to punish since the Helper was generous."
                    reward = 0
                else:
                    event = f"{player_name} pays ${self.config.punishment_cost} to reduce Helper's earnings by ${self.config.punishment_impact}."
                    reward = -self.config.punishment_cost
                    self.helper_punished = True
            else:
                event = f"{player_name} chooses not to punish the Helper."
                reward = 0
                self.helper_punished = False

        # Log the event
        self.results_log.append({
            'round': self.round,
            'time': str(self.clock.now()),
            'player': player_name,
            'action': action,
            'event': event,
            'helped': helped,  # 'helped' is now always defined
            'reward': reward
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

    def save_results(self, filename: str):
        """Save results to JSON file."""
        results = {
            'config': self.config.__dict__,
            'events': self.results_log
        }
        
        # Create results directory if it doesn't exist
        pathlib.Path('results').mkdir(exist_ok=True)
        
        with open(f'results/{filename}.json', 'w') as f:
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
    helper = build_tpp_agent(
        config=formative_memories.AgentConfig(
            name="Helper",
            goal="As the Helper, decide whether to send $5.0 to the Recipient or keep it. State your decision clearly as 'I choose to send $5.0 to the Recipient' or 'I choose to keep the $5.0.'",
            extras={'main_character': True}
        ),
        model=model,
        memory=associative_memory.AssociativeMemory(embedder),
        clock=clock,
        is_public=public_condition,
        persona=personas[0],
    )

    recipient = build_tpp_agent(
        config=formative_memories.AgentConfig(
            name="Recipient",
            goal="As the Recipient, decide how much money to send back to the Helper after receiving the tripled amount. State your decision clearly as 'I choose to send back $X to the Helper.'",
            extras={'main_character': True}
        ),
        model=model,
        memory=associative_memory.AssociativeMemory(embedder),
        clock=clock,
        is_public=public_condition,
        persona=personas[1],
    )

    punisher = build_tpp_agent(
        config=formative_memories.AgentConfig(
            name="Punisher",
            goal="As the Punisher, decide whether to punish the Helper based on their action. State your decision clearly as 'I choose to punish the Helper' or 'I choose not to punish the Helper.'",
            extras={'main_character': True}
        ),
        model=model,
        memory=associative_memory.AssociativeMemory(embedder),
        clock=clock,
        is_public=public_condition,
        persona=personas[2],
    )

    # Create game master with agents
    gm = TPPGameMaster(
        config=config,
        model=model,
        embedder=embedder,
        clock=clock,
        measurements=measurements,
        helper_agent=helper,
        recipient_agent=recipient,
        punisher_agent=punisher
    )

    # Initialize agents with scenario context
    helper.observe(config.scenario_description)
    recipient.observe(config.scenario_description)
    punisher.observe(config.scenario_description)

    # Run experiment rounds
    for _ in range(config.n_rounds):
        print(f"Round {gm.round + 1}")
        helper_event, recipient_event, punisher_event = gm.run_tpp_round()
        gm.step()

    # Save results
    condition = "public" if public_condition else "anonymous"
    gm.save_results(f"tpp_experiment_{condition}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")


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

def test_tpp_hypothesis(
    model: language_model.LanguageModel,
    embedder,
    n_rounds: int = 3,
):
    """Run complete TPP experiment testing public vs anonymous hypothesis."""
    
    # Initialize clock
    clock = game_clock.MultiIntervalClock(
        start=datetime.datetime(2024, 1, 1),
        step_sizes=[datetime.timedelta(minutes=5)],
    )
    
    # Initialize measurements
    measurements = measurements_lib.Measurements()
    
    # Create experiment config
    config = TPPScenarioConfig(n_rounds=n_rounds)

    # Sample three distinct personas for the roles
    helper_persona, recipient_persona, punisher_persona = assign_personas()
    personas = [helper_persona, recipient_persona, punisher_persona]
    
    print("Running public condition...")
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
    public_file = max(results_dir.glob('tpp_experiment_public_*.json'))
    anonymous_file = max(results_dir.glob('tpp_experiment_anonymous_*.json'))
    
    analysis = analyze_results(public_file.name, anonymous_file.name)
    
    print("\nResults Analysis:")
    print(f"Public condition punishment rate: {analysis['public']['punishment_rate']:.2%}")
    print(f"Anonymous condition punishment rate: {analysis['anonymous']['punishment_rate']:.2%}")
    print(f"\nPublic condition helping rate: {analysis['public']['helping_rate']:.2%}")
    print(f"Anonymous condition helping rate: {analysis['anonymous']['helping_rate']:.2%}")