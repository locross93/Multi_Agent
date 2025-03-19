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
        
        super().__init__(
            pre_act_key=f"\nContribution Decision",
            question=question,
            answer_prefix=answer_prefix,
            add_to_memory=True,
            memory_tag="[contribution decision]",
            components=components,
            **kwargs,
        )

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

    return agent