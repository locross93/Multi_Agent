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
            f"{persona.background}. Your personality is characterized as {persona.traits}. "
            f"Based on this background and your past actions, how would you describe your approach "
            f"to economic games like the public goods game? Are you more focused on maximizing group welfare or your own payoff?"
        )
        answer_prefix = f"{agent_name}, being {persona.name}, is someone who "
        
        super().__init__(
            pre_act_key=f"\nCharacter Assessment",
            question=question,
            answer_prefix=answer_prefix,
            add_to_memory=False,
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
            add_to_memory=False,
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
            add_to_memory=False,
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
            add_to_memory=False,
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

class StrategyReflection(question_of_recent_memories.QuestionOfRecentMemories):
    """Component to reason about the other players' personalities and likely behavior."""
    
    def __init__(self, agent_name: str, **kwargs):
        question = (
            f"Think step by step about the best strategy for you to maximize your long term earnings in the public goods game."
            f"Focus on selfishly maximizing your own earnings."
            f"If it's past round 1, think about how you should adjust your strategy based on your observations so far."
            f"Think about what the most successful players in the game have done so far and what you can learn from them."
            f"Could you have made a better decision to maximize your personal earnings in the past based on the information available to you?"
            f"If so, what will you do differently in the future?"
        )
        answer_prefix = "Based on what we know, "
        super().__init__(
            pre_act_key="\nStrategy Reflection",
            question=question,
            answer_prefix=answer_prefix,
            add_to_memory=False,
            memory_tag="[strategy_reflection]",
            components={
                'Observation': '\nObservation',
                'ObservationSummary': '\nRecent context',
            },
            num_memories_to_retrieve=10,
            terminators=None,
            **kwargs,
        )

class EmotionReflection(question_of_recent_memories.QuestionOfRecentMemories):
    """Component to reflect on the agent's emotional state in the current game situation."""
    def __init__(self, agent_name: str, persona: Persona, **kwargs):
        self.persona = persona
        question = (
            f"As {persona.name}, reflect on how you're feeling emotionally about the current situation"
        )
        answer_prefix = f"{persona.name} is feeling "
        
        super().__init__(
            pre_act_key="\nEmotional State",
            question=question,
            answer_prefix=answer_prefix,
            add_to_memory=False,
            memory_tag="[emotion_reflection]",
            components={
                'Observation': '\nObservation',
                'ObservationSummary': '\nRecent context',
                'PersonalityReflection': '\nCharacter Assessment',
                'TheoryOfMind': '\nTheory of Mind Analysis',
                'TheoryOfMind2': '\nTheory of Mind Analysis 2',
                'SituationAssessment': '\nSituation Analysis'
            },
            num_memories_to_retrieve=10,
            **kwargs,
        )


class DecisionReflection(QuestionOfRecentMemoriesWithActionSpec):
    def __init__(self, agent_name: str, persona: Persona, has_theory_of_mind=True, has_emotion_reflection=False, has_strategy_reflection=False, **kwargs):
        self.persona = persona
        # Define components based on available capabilities
        components = {
            'Observation': '\nObservation',
            'ObservationSummary': '\nRecent context',
            'PersonalityReflection': '\nCharacter Assessment',
        }
        
        # Only add TheoryOfMind if the agent has that capability
        if has_theory_of_mind:
            components['TheoryOfMind'] = '\nTheory of Mind Analysis'
            components['TheoryOfMind2'] = '\nTheory of Mind Analysis 2'

        if has_emotion_reflection:
            components['EmotionReflection'] = '\nEmotional State'

        if has_strategy_reflection:
            components['StrategyReflection'] = '\nStrategy Reflection'

        super().__init__(
            pre_act_key="\nDecision Reflection",
            #question="Think step by step and reflect what you should do next in the current game situation: {question}.",
            #question = "Based on the above context about the situation and {agent_name}, think step by step about what they will decide in the current situation: {question}.",
            question="Based on the above context about the situation and "+persona.name+", think step by step about what "+persona.name+" will decide in the current situation: {question}.",
            #question="Based on the above context about the situation and "+persona.name+", think step by step about what "+persona.name+" will decide in the current situation to maximize their long term earnings: {question}.",
            answer_prefix=f"{persona.name} will ",
            add_to_memory=False,
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
    has_strategy_reflection = True

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
            persona=persona,
            model=model,
            logging_channel=measurements.get_channel('EmotionReflection').on_next,
        )
        components['EmotionReflection'] = emotion_reflection

    if has_strategy_reflection:
        strategy_reflection = StrategyReflection(
            agent_name=agent_name,
            model=model,
            logging_channel=measurements.get_channel('StrategyReflection').on_next,
        )
        components['StrategyReflection'] = strategy_reflection
    # New DecisionReflection component
    decision_reflection = DecisionReflection(
        agent_name=agent_name,
        persona=persona,
        model=model,
        has_theory_of_mind=has_theory_of_mind,
        has_emotion_reflection=has_emotion_reflection,
        has_strategy_reflection=has_strategy_reflection,
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

    components_keys = list(components.keys())
    return agent, components_keys