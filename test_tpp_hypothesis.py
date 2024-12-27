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

from concordia_trust_game import build_tpp_agent

@dataclass
class TPPScenarioConfig:
    """Configuration for TPP experiment scenario."""
    helper_endowment: float = 10.0
    helping_amount: float = 5.0
    punishment_cost: float = 2.0
    punishment_impact: float = 6.0
    n_rounds: int = 20
    scenario_description: str = """You are in a room where a research experiment is taking place.
    There are three roles: Helpers who can share money with Recipients, and Punishers who can 
    pay to reduce a Helper's earnings if they are selfish."""

class TPPGameMaster(game_master.GameMaster):
    """Game master for TPP experiment."""

    def __init__(
        self,
        config: TPPScenarioConfig,
        model: language_model.LanguageModel,
        clock: game_clock.MultiIntervalClock,
        measurements: measurements_lib.Measurements,
    ):
        super().__init__(model=model)
        self.config = config
        self.clock = clock
        self.measurements = measurements
        self.round = 0
        self.results_log = []

    def generate_event(self, player_name: str, action: str) -> str:
        """Generate event outcomes based on player actions."""
        if "help" in action.lower():
            helped = True
            event = f"{player_name} shares ${self.config.helping_amount} with the Recipient."
            reward = -self.config.helping_amount
        elif "punish" in action.lower() and "selfish" in self._last_helper_event.lower():
            helped = False
            event = f"{player_name} pays ${self.config.punishment_cost} to reduce selfish Helper's earnings by ${self.config.punishment_impact}."
            reward = -self.config.punishment_cost
        else:
            helped = False
            event = f"{player_name} takes no action."
            reward = 0

        # Log the event
        self.results_log.append({
            'round': self.round,
            'time': str(self.clock.now()),
            'player': player_name,
            'action': action,
            'event': event,
            'helped': helped,
            'reward': reward
        })

        return event

    def run_tpp_round(self, helper_agent: entity_agent_with_logging.EntityAgentWithLogging,
                     punisher_agent: entity_agent_with_logging.EntityAgentWithLogging) -> Tuple[str, str]:
        """Run one round of the TPP game."""
        self.round += 1
        
        # Helper's turn
        helper_obs = f"Round {self.round}: You are the Helper. You have ${self.config.helper_endowment} and can choose to share ${self.config.helping_amount} with the Recipient."
        helper_agent.observe(helper_obs)
        helper_action = helper_agent.act()
        helper_event = self.generate_event(helper_agent.name, helper_action)
        self._last_helper_event = helper_event

        # Punisher's turn 
        punisher_obs = f"Round {self.round}: You observe that {helper_event}"
        punisher_agent.observe(punisher_obs)
        punisher_action = punisher_agent.act()
        punisher_event = self.generate_event(punisher_agent.name, punisher_action)

        return helper_event, punisher_event

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
    public_condition: bool = True,
) -> None:
    """Run TPP experiment with logging."""
    
    # Initialize clock and measurements
    clock = game_clock.MultiIntervalClock(
        start_time=datetime.datetime(2024, 1, 1),
        intervals=[datetime.timedelta(minutes=5)],
    )
    
    measurements = measurements_lib.Measurements()
    config = TPPScenarioConfig()

    # Create game master
    gm = TPPGameMaster(
        config=config,
        model=model,
        clock=clock,
        measurements=measurements,
    )

    # Initialize agents
    helper = build_tpp_agent(
        config=formative_memories.AgentConfig(
            name="Helper",
            goal="Maximize personal earnings while considering moral implications.",
            extras={'main_character': True}
        ),
        model=model,
        memory=associative_memory.AssociativeMemory(embedder),
        clock=clock,
        is_public=public_condition,
    )

    punisher = build_tpp_agent(
        config=formative_memories.AgentConfig(
            name="Punisher",
            goal="Maintain fairness and justice in the experiment, considering the costs.",
            extras={'main_character': True}
        ),
        model=model,
        memory=associative_memory.AssociativeMemory(embedder),
        clock=clock,
        is_public=public_condition,
    )

    # Initialize agents with scenario context
    helper.observe(config.scenario_description)
    punisher.observe(config.scenario_description)

    # Run experiment rounds
    for _ in range(config.n_rounds):
        helper_event, punisher_event = gm.run_tpp_round(helper, punisher)
        clock.advance_time()

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

def test_tpp_hypothesis():
    """Run complete TPP experiment testing public vs anonymous hypothesis."""
    
    # Run public condition
    print("Running public condition...")
    run_tpp_experiment(model=model, embedder=embedder, public_condition=True)
    
    # Run anonymous condition
    print("Running anonymous condition...")
    run_tpp_experiment(model=model, embedder=embedder, public_condition=False)
    
    # Find most recent result files
    results_dir = pathlib.Path('results')
    public_file = max(results_dir.glob('tpp_experiment_public_*.json'))
    anonymous_file = max(results_dir.glob('tpp_experiment_anonymous_*.json'))
    
    # Analyze results
    analysis = analyze_results(public_file.name, anonymous_file.name)
    
    print("\nResults Analysis:")
    print(f"Public condition punishment rate: {analysis['public']['punishment_rate']:.2%}")
    print(f"Anonymous condition punishment rate: {analysis['anonymous']['punishment_rate']:.2%}")
    print(f"\nPublic condition helping rate: {analysis['public']['helping_rate']:.2%}")
    print(f"Anonymous condition helping rate: {analysis['anonymous']['helping_rate']:.2%}")

if __name__ == "__main__":
    test_tpp_hypothesis()