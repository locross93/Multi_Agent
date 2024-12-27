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

class PersonalityReflection(question_of_recent_memories.QuestionOfRecentMemories):
    """Component to assess agent's moral character and trustworthiness."""
    
    def __init__(self, agent_name: str, **kwargs):
        question = "Given the above, what kind of moral character does {agent_name} have?"
        answer_prefix = "{agent_name} is a person who "
        super().__init__(
            pre_act_key=f"\nQuestion: {question}\nAnswer",
            question=question,
            answer_prefix=answer_prefix,
            add_to_memory=True,
            memory_tag="[character reflection]",
            components={}, 
            **kwargs,
        )

class SituationAssessment(question_of_recent_memories.QuestionOfRecentMemories):
    """Component to assess the current moral situation."""
    
    def __init__(self, agent_name: str, **kwargs):
        question = "Given the statements above, what moral decision does {agent_name} face right now?"
        answer_prefix = "{agent_name} is in a situation where "
        super().__init__(
            pre_act_key=f"\nQuestion: {question}\nAnswer",
            question=question, 
            answer_prefix=answer_prefix,
            add_to_memory=True,
            memory_tag="[situation assessment]",
            components={
                'Observation': '\nObservation',
                'ObservationSummary': '\nRecent context'
            },
            **kwargs,
        )

class PunishmentDecision(question_of_recent_memories.QuestionOfRecentMemories):
    """Component to decide whether to punish wrongdoing."""
    
    def __init__(self, agent_name: str, is_public: bool = True, **kwargs):
        context = "public" if is_public else "private"
        question = f"What should {agent_name} do in response to this wrongdoing? Note: This is a {context} situation where others will {'see' if is_public else 'not see'} the decision."
        answer_prefix = f"{agent_name} decides to "
        
        super().__init__(
            pre_act_key=f"\nQuestion: {question}\nAnswer",
            question=question,
            answer_prefix=answer_prefix,
            add_to_memory=True,
            memory_tag="[punishment decision]",
            components={
                'PersonalityReflection': '\nMoral character',
                'SituationAssessment': '\nMoral situation',
            },
            **kwargs,
        )

def build_tpp_agent(
    config: formative_memories.AgentConfig,
    model: language_model.LanguageModel,
    memory: associative_memory.AssociativeMemory,
    clock: game_clock.MultiIntervalClock,
    is_public: bool = True,
) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build a TPP agent with components for moral reasoning."""
    
    agent_name = config.name
    raw_memory = legacy_associative_memory.AssociativeMemoryBank(memory)
    measurements = measurements_lib.Measurements()

    # Core components
    observation = agent_components.observation.Observation(
        clock_now=clock.now,
        timeframe=clock.get_step_size(),  # Add timeframe from clock
        pre_act_key='\nObservation',
        logging_channel=measurements.get_channel('Observation').on_next,
    )
    
    obs_summary = agent_components.observation.ObservationSummary(
        model=model,
        clock_now=clock.now,
        timeframe_delta_from=timedelta(hours=4),  # Add timeframe deltas
        timeframe_delta_until=timedelta(hours=0),
        pre_act_key='Recent context',
        logging_channel=measurements.get_channel('ObservationSummary').on_next,
    )

    # TPP specific components
    personality = PersonalityReflection(
        agent_name=agent_name,
        model=model,
        logging_channel=measurements.get_channel('PersonalityReflection').on_next,
    )
    
    situation = SituationAssessment(
        agent_name=agent_name,
        model=model,
        logging_channel=measurements.get_channel('SituationAssessment').on_next,
    )
    
    decision = PunishmentDecision(
        agent_name=agent_name,
        is_public=is_public,
        model=model,
        logging_channel=measurements.get_channel('PunishmentDecision').on_next,
    )

    # Assemble components
    entity_components = [observation, obs_summary, personality, situation, decision]
    components = {
        component.__class__.__name__: component 
        for component in entity_components
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
        agent_name=agent_name,
        act_component=act_component,
        context_components=components,
        component_logging=measurements,
    )

    return agent