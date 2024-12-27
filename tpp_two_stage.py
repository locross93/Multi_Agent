# agent_components.py
from datetime import datetime, timedelta
from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import associative_memory, formative_memories
from concordia.memory_bank import legacy_associative_memory
from concordia.components import agent as agent_components
from concordia.components.agent import question_of_recent_memories, memory_component
from concordia.typing import entity_component
from concordia.language_model import language_model
from concordia.clocks import game_clock
from concordia.utils import measurements as measurements_lib

class TrustReflection(question_of_recent_memories.QuestionOfRecentMemories):
    """Component to assess trust-relevant information about interaction partners."""
    
    def __init__(self, agent_name: str, **kwargs):
        question = f"Given the above, what does {agent_name} know about their interaction partner's trustworthiness?"
        answer_prefix = f"Based on their knowledge, {agent_name} believes their partner "
        super().__init__(
            pre_act_key=f"\nQuestion: {question}\nAnswer",
            question=question,
            answer_prefix=answer_prefix,
            add_to_memory=True,
            memory_tag="[trust reflection]",
            components={
                'Observation': '\nRecent observations',
                'ObservationSummary': '\nKnowledge of partner'
            },
            **kwargs,
        )

class SelfInterestAssessment(question_of_recent_memories.QuestionOfRecentMemories):
    """Component to assess potential gains and losses."""
    
    def __init__(self, agent_name: str, **kwargs):
        question = f"What are the potential gains and losses {agent_name} faces in this situation?"
        answer_prefix = f"{agent_name} faces "
        super().__init__(
            pre_act_key=f"\nQuestion: {question}\nAnswer",
            question=question,
            answer_prefix=answer_prefix,
            add_to_memory=True,
            memory_tag="[self-interest assessment]",
            components={
                'Observation': '\nCurrent situation',
            },
            **kwargs,
        )

class MoralDecision(question_of_recent_memories.QuestionOfRecentMemories):
    """Component for making moral/trust decisions."""
    
    def __init__(self, agent_name: str, is_public: bool = True, **kwargs):
        context = "public" if is_public else "private"
        question = f"What should {agent_name} do in this situation? Note: This is a {context} situation where others will {'see' if is_public else 'not see'} the decision."
        answer_prefix = f"{agent_name} decides to "
        super().__init__(
            pre_act_key=f"\nQuestion: {question}\nAnswer",
            question=question,
            answer_prefix=answer_prefix,
            add_to_memory=True,
            memory_tag="[decision]",
            components={
                'TrustReflection': '\nTrust Assessment',
                'SelfInterestAssessment': '\nStakes Assessment',
            },
            **kwargs,
        )

def build_trust_agent(
    config: formative_memories.AgentConfig,
    model: language_model.LanguageModel,
    memory: associative_memory.AssociativeMemory,
    clock: game_clock.MultiIntervalClock,
    is_public: bool = True,
    role: str = "punisher",  # Can be "punisher", "helper", "recipient", "chooser", "signaler"
) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build an agent for the trust game experiments."""
    
    agent_name = config.name
    raw_memory = legacy_associative_memory.AssociativeMemoryBank(memory)
    measurements = measurements_lib.Measurements()

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

    # Trust game specific components
    trust_reflection = TrustReflection(
        agent_name=agent_name,
        model=model,
        logging_channel=measurements.get_channel('TrustReflection').on_next,
    )
    
    self_interest = SelfInterestAssessment(
        agent_name=agent_name,
        model=model,
        logging_channel=measurements.get_channel('SelfInterestAssessment').on_next,
    )
    
    moral_decision = MoralDecision(
        agent_name=agent_name,
        is_public=is_public,
        model=model,
        logging_channel=measurements.get_channel('MoralDecision').on_next,
    )

    # Assemble components
    entity_components = [observation, obs_summary, trust_reflection, self_interest, moral_decision]
    components = {
        component.__class__.__name__: component 
        for component in entity_components
    }
    
    # Add memory component
    components[memory_component.DEFAULT_MEMORY_COMPONENT_NAME] = memory_component.MemoryComponent(raw_memory)

    # Create act component
    act_component = agent_components.concat_act_component.ConcatActComponent(
        model=model,
        clock=clock,
        component_order=list(components.keys()),
        logging_channel=measurements.get_channel('ActComponent').on_next,
    )

    # Build agent
    agent = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=agent_name,
        act_component=act_component,
        context_components=components,
        component_logging=measurements,
    )

    return agent

@dataclass
class TPPTrustGameConfig:
    """Configuration for two-stage TPP trust game experiment."""
    # Stage 1 - TPP Game
    helper_endowment_s1: float = 10.0
    helping_amount_s1: float = 5.0
    punishment_cost: float = 2.0
    punishment_impact: float = 6.0
    
    # Stage 2 - Trust Game
    chooser_endowment: float = 10.0
    multiplier: float = 3.0  # Amount sent gets tripled
    
    n_rounds: int = 1  # Usually just one round per stage
    
    stage1_description: str = """Stage 1: Trust Game with Punishment
    Brock (Helper) has received $10 and can choose to share $5 with Lisa (Recipient).
    Any shared amount would be tripled. Eli (Punisher) can pay $2 to reduce Brock's earnings by $6 if Brock acts selfishly."""
    
    stage2_description_public: str = """Stage 2: Trust Game
    Lisa has $10 and can choose to send any amount to Eli. Whatever is sent will be tripled.
    Eli can then choose how much of the tripled amount to return to Lisa.
    Everyone knows about Eli's punishment decision in Stage 1."""
    
    stage2_description_private: str = """Stage 2: Trust Game
    Lisa has $10 and can choose to send any amount to Eli. Whatever is sent will be tripled.
    Eli can then choose how much of the tripled amount to return to Lisa."""

class TwoStageTrustGameMaster(game_master.GameMaster):
    """Game master for two-stage trust game experiment."""

    def __init__(
        self,
        config: TPPTrustGameConfig,
        model: language_model.LanguageModel,
        embedder,
        clock: game_clock.MultiIntervalClock,
        measurements: measurements_lib.Measurements,
        stage1_punisher: entity_agent_with_logging.EntityAgentWithLogging,  # Eli
        stage1_helper: entity_agent_with_logging.EntityAgentWithLogging,    # Brock
        stage1_recipient: entity_agent_with_logging.EntityAgentWithLogging, # Lisa
        public_condition: bool = True,
    ):
        # Create memory for game master
        memory = associative_memory.AssociativeMemory(embedder)
        
        # Store all agents
        self.stage1_punisher = stage1_punisher  # Eli
        self.stage1_helper = stage1_helper      # Brock
        self.stage1_recipient = stage1_recipient # Lisa
        
        # Initialize base GameMaster
        super().__init__(
            model=model,
            memory=memory,
            clock=clock,
            players=[stage1_punisher, stage1_helper, stage1_recipient],
            action_spec=DEFAULT_ACTION_SPEC,
        )
        
        self.config = config
        self.public_condition = public_condition
        self.measurements = measurements
        self.stage = 1
        self.results_log = []

    def run_stage1(self) -> str:
        """Run stage 1 - TPP trust game with forced selfish behavior."""
        print("\nRunning Stage 1...")
        
        # Force Brock to act selfishly
        helper_event = f"Brock (Helper) decides to keep the ${self.config.helper_endowment_s1} and share nothing with Lisa."
        self.stage1_recipient.observe(helper_event)
        self.stage1_punisher.observe(helper_event)
        
        # Let Eli decide whether to punish
        punisher_obs = f"You are Eli. You observe that {helper_event}. You can pay ${self.config.punishment_cost} to reduce Brock's earnings by ${self.config.punishment_impact}."
        self.stage1_punisher.observe(punisher_obs)
        punisher_action = self.stage1_punisher.act()
        
        # Process punishment decision
        if "punish" in punisher_action.lower():
            outcome = f"Eli pays ${self.config.punishment_cost} to punish Brock, reducing Brock's earnings by ${self.config.punishment_impact}."
        else:
            outcome = "Eli decides not to punish Brock."
            
        # Everyone observes the outcome
        for agent in [self.stage1_helper, self.stage1_recipient, self.stage1_punisher]:
            agent.observe(outcome)
            
        return outcome

    def run_stage2(self, stage1_outcome: str):
        """Run stage 2 - Trust game between Lisa (Chooser) and Eli (Signaler)."""
        print("\nRunning Stage 2...")
        
        # Set up stage 2 context
        if self.public_condition:
            lisa_context = f"""You are Lisa. You have ${self.config.chooser_endowment} to potentially share with Eli.
            Any amount you send will be tripled. Eli can then decide how much to return to you.
            You know that in Stage 1, {stage1_outcome}"""
        else:
            lisa_context = f"""You are Lisa. You have ${self.config.chooser_endowment} to potentially share with Eli.
            Any amount you send will be tripled. Eli can then decide how much to return to you."""
            
        # Lisa decides how much to send
        self.stage1_recipient.observe(lisa_context)
        lisa_action = self.stage1_recipient.act()
        
        # Process Lisa's decision
        # [Rest of stage 2 implementation...]