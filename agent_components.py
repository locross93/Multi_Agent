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

# class PersonalityReflection(question_of_recent_memories.QuestionOfRecentMemories):
#     """Component to assess agent's moral character and trustworthiness."""
    
#     def __init__(self, agent_name: str, **kwargs):
#         question = f"Based on past actions and decisions, what are {agent_name}'s key traits and values?"
#         answer_prefix = f"{agent_name} is someone who "
#         super().__init__(
#             pre_act_key=f"\nCharacter Assessment",
#             question=question,
#             answer_prefix=answer_prefix,
#             add_to_memory=True,
#             memory_tag="[character reflection]",
#             components={}, 
#             **kwargs,
#         )

    # def _make_pre_act_value(self) -> str:
    #     print(f"\n=== {self.__class__.__name__} ===")
    #     print("Component:", self.get_pre_act_key())
        
    #     # Get memories
    #     memory = self.get_entity().get_component(
    #         self._memory_component_name,
    #         type_=memory_component.MemoryComponent)
        
    #     mems = '\n'.join([
    #         mem.text for mem in memory.retrieve(
    #             scoring_fn=legacy_associative_memory.RetrieveRecent(add_time=True),
    #             limit=self._num_memories_to_retrieve,
    #         )
    #     ])
        
    #     # Create and log the full prompt
    #     prompt = interactive_document.InteractiveDocument(self._model)
    #     prompt.statement(f'Recent memories:\n{mems}')
        
    #     print("\nFull Prompt:")
    #     print("------------")
    #     print(prompt.view().text())
    #     print(f"Question: {self._question}")
    #     print("------------")
        
    #     result = prompt.open_question(
    #         self._question,
    #         max_tokens=1000,
    #         answer_prefix=self._answer_prefix,
    #     )
        
    #     print("LLM Response:", result)
    #     print("=== End ===\n")
    #     return result

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
            components={
                'Observation': '\nObservation',
                'ObservationSummary': '\nRecent context'
            },
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
                'PersonalityReflection': '\nCharacter Assessment',
                'SituationAssessment': '\nSituation Analysis',
            },
            **kwargs,
        )

class PunisherDecision(question_of_recent_memories.QuestionOfRecentMemories):
    """Component for Punisher's decision-making."""
    
    def __init__(self, agent_name: str, is_public: bool = True, **kwargs):
        context = "public" if is_public else "private"
        question = (
            f"As the Punisher observing the interaction, {agent_name} can spend $2.0 to reduce "
            f"the Helper's payoff by $6.0. This is a {context} decision. "
            f"What will {agent_name} do?"
        )
        answer_prefix = f"{agent_name} "
        
        super().__init__(
            pre_act_key=f"\nPunisher Decision",
            question=question,
            answer_prefix=answer_prefix,
            add_to_memory=True,
            memory_tag="[punisher decision]",
            components={
                'PersonalityReflection': '\nCharacter Assessment',
                'SituationAssessment': '\nSituation Analysis',
            },
            **kwargs,
        )

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
        agent_name=agent_role,
        act_component=act_component,
        context_components=components,
        component_logging=measurements,
    )

    return agent