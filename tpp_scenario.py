from dataclasses import dataclass
import re
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
PUNISHER_ACTION_SPEC = choice_action_spec(
    call_to_action=(
        "Do you want to spend $2.0 to reduce the Helper's payoff by $6.0?"
    ),
    options=["Yes, I choose to punish the Helper", "No, I choose not to punish the Helper"],
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
    n_rounds: int = 1
    scenario_description: str = """You are in a room where a two-stage research experiment is taking place.
    There are four players and you will be randomly assigned to play as Player 1, 2, 3, or 4. Every player will first get an endowment of $5.
    In Stage 1, Player 1 and Player 2 will play an economic game. 
    Player 1 is given an additional $10 and can choose to send any portion of this money to Player 2. 
    Any amount sent to Player 2 will be tripled by the experimenter. 
    For example, if Player 1 sends $5, Player 2 will receive $15. 
    Player 2 can then choose to send any amount of this tripled money back to Player 1. 
    While this interaction occurs, Player 3 and Player 4 will observe the decisions made. 
    After Player 1 and Player 2 complete their interaction, Player 3 will have the option to spend $2 of their endowment to reduce Player 1's payoff by $6 if they want to punish Player 1. 
    Player 4 will observe whether Player 3 chooses to punish Player 1 or not.
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
        self.chooser_agent.observe(punisher_obs)

        # Punisher's turn
        # Use the multiple choice action spec for Punisher
        punisher_action = self.signaller_agent.act(PUNISHER_ACTION_SPEC)
        #punisher_action = self.signaller_agent.act(TPP_ACTION_SPEC)
        punisher_event = self.generate_event(self.signaller_agent.name, punisher_action)
        # add actions to both agents observations
        self.signaller_agent.observe(punisher_event)
        self.chooser_agent.observe(punisher_event)

        return punisher_event

    def run_tpp_round2(self) -> Tuple[str, str]:
        """Run stage 2 of the TPP game."""
        self.round += 1

        # Helper's turn
        helper_spec = free_action_spec(
            call_to_action=STAGE2_HELPER_ACTION_SPEC.call_to_action.format(
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
            recipient_spec = free_action_spec(
                call_to_action=STAGE2_RECIPIENT_ACTION_SPEC.call_to_action.format(
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

        breakpoint()

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
            'reward': reward
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
    )

    # Initialize agents with scenario context

    signaller.observe(config.scenario_description)
    chooser.observe(config.scenario_description)

    # run stage 1   
    punisher_event = gm.run_tpp_round1()

    # run stage 2
    helper_event, recipient_event = gm.run_tpp_round2()

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
    signaller_persona, chooser_persona = assign_personas(n=2)
    personas = [signaller_persona, chooser_persona]
    
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
    
    # print("Running anonymous condition...")
    # run_tpp_experiment(
    #     model=model,
    #     embedder=embedder,
    #     clock=clock,
    #     measurements=measurements,
    #     config=config,
    #     public_condition=False,
    #     personas=personas,
    # )
    
    # Analyze results
    results_dir = pathlib.Path('results')
    public_file = max(results_dir.glob('tpp_experiment_public_*.json'))
    #anonymous_file = max(results_dir.glob('tpp_experiment_anonymous_*.json'))
    
    # analysis = analyze_results(public_file.name, anonymous_file.name)
    
    # print("\nResults Analysis:")
    # print(f"Public condition punishment rate: {analysis['public']['punishment_rate']:.2%}")
    # #print(f"Anonymous condition punishment rate: {analysis['anonymous']['punishment_rate']:.2%}")
    # print(f"\nPublic condition helping rate: {analysis['public']['helping_rate']:.2%}")
    # #print(f"Anonymous condition helping rate: {analysis['anonymous']['helping_rate']:.2%}")