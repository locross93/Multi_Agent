import sentence_transformers

from concordia import typing
from concordia.typing import entity

from concordia.associative_memory import associative_memory
from concordia.language_model import gpt_model
from concordia.language_model import language_model
from concordia.components.agent import action_spec_ignored
from concordia.memory_bank import legacy_associative_memory
from concordia.agents import entity_agent
from concordia.typing import entity_component
from concordia.components.agent import memory_component

import collections
import json

# The memory will use a sentence embedder for retrievel, so we download one from
# Hugging Face.
_embedder_model = sentence_transformers.SentenceTransformer(
    'sentence-transformers/all-mpnet-base-v2')
embedder = lambda x: _embedder_model.encode(x, show_progress_bar=False)

api_key_path = '/ccn2/u/locross/RPS_ToM/llm_plan/lc_api_key.json'
OPENAI_KEYS = json.load(open(api_key_path, 'r'))
GPT_API_KEY = OPENAI_KEYS['API_KEY']
GPT_MODEL_NAME = 'gpt-4o' #@param {type: 'string'}

if not GPT_API_KEY:
  raise ValueError('GPT_API_KEY is required.')

model = gpt_model.GptLanguageModel(api_key=GPT_API_KEY,
                                   model_name=GPT_MODEL_NAME)

def make_prompt(deque: collections.deque[str]) -> str:
  """Makes a string prompt by joining all observations, one per line."""
  return "\n".join(deque)


# class SimpleLLMAgent(entity.Entity):

#   def __init__(self, model: language_model.LanguageModel):
#     self._model = model
#     # Container (circular queue) for observations.
#     self._memory = collections.deque(maxlen=5)

#   @property
#   def name(self) -> str:
#     return 'Alice'

#   def act(self, action_spec=entity.DEFAULT_ACTION_SPEC) -> str:
#     prompt = make_prompt(self._memory)
#     print(f"*****\nDEBUG: {prompt}\n*****")
#     return self._model.sample_text(
#         "You are a person.\n"
#         f"Your name is {self.name} and your recent observations are:\n"
#         f"{prompt}\nWhat should you do next?")

#   def observe(
#       self,
#       observation: str,
#   ) -> None:
#     # Push a new observation into the memory, if there are too many, the oldest
#     # one will be automatically dropped.
#     self._memory.append(observation)

# # agent = SimpleLLMAgent(model)       

# # agent.observe("You absolutely hate apples and would never willingly eat them.")
# # agent.observe("You don't particularly like bananas.")
# # # Only the next 5 observations will be kept, pushing out critical information!
# # #agent.observe("You are in a room.")
# # #agent.observe("The room has only a table in it.")
# # agent.observe("On the table there are two fruits: an apple and a banana.")
# # agent.observe("The apple is shinny red and looks absolutely irresistible!")
# # agent.observe("The banana is slightly past its prime.")
# # response = agent.act()

# # print(response)

# def make_prompt_associative_memory(
#     memory: associative_memory.AssociativeMemory) -> str:
#   """Makes a string prompt by joining all observations, one per line."""
#   recent_memories_list = memory.retrieve_recent(5)
#   recent_memories_set = set(recent_memories_list)
#   recent_memories = "\n".join(recent_memories_set)

#   relevant_memories_list = []
#   for recent_memory in recent_memories_list:
#     # Retrieve 3 memories that are relevant to the recent memory.
#     relevant = memory.retrieve_associative(recent_memory, 3, add_time=False)
#     for mem in relevant:
#       # Make sure that we only add memories that are _not_ already in the recent
#       # ones.
#       if mem not in recent_memories_set:
#         relevant_memories_list.append(mem)

#   relevant_memories = "\n".join(relevant_memories_list)
#   return (
#       f"Your recent memories are:\n{recent_memories}\n"
#       f"Relevant memories from your past:\n{relevant_memories}\n"
#   )


# class SimpleLLMAgentWithAssociativeMemory(entity.Entity):

#   def __init__(self, model: language_model.LanguageModel, embedder):
#     self._model = model
#     # The associative memory of the agent. It uses a sentence embedder to
#     # retrieve on semantically relevant memories.
#     self._memory = associative_memory.AssociativeMemory(embedder)

#   @property
#   def name(self) -> str:
#     return 'Alice'

#   def act(self, action_spec=entity.DEFAULT_ACTION_SPEC) -> str:
#     prompt = make_prompt_associative_memory(self._memory)
#     print(f"*****\nDEBUG: {prompt}\n*****")
#     return self._model.sample_text(
#         "You are a person.\n"
#         f"Your name is {self.name}.\n"
#         f"{prompt}\n"
#         "What should you do next?")

#   def observe(
#       self,
#       observation: str,
#   ) -> None:
#     # Push a new observation into the memory, if there are too many, the oldest
#     # one will be automatically dropped.
#     self._memory.add(observation)

# # agent = SimpleLLMAgentWithAssociativeMemory(model, embedder)

# # agent.observe("You absolutely hate apples and would never willingly eat them.")
# # agent.observe("You don't particularly like bananas.")
# # # Only the next 5 observations will be retrieved as "recent memories"
# # agent.observe("You are in a room.")
# # agent.observe("The room has only a table in it.")
# # agent.observe("On the table there are two fruits: an apple and a banana.")
# # agent.observe("The apple is shinny red and looks absolutely irresistible!")
# # agent.observe("The banana is slightly past its prime.")
# # response2 = agent.act()

# # print(response2)


class Observe(entity_component.ContextComponent):

  def pre_observe(self, observation: str) -> None:
    self.get_entity().get_component('memory').add(observation, {})


class RecentMemories(entity_component.ContextComponent):

  def pre_act(self, action_spec) -> None:
    recent_memories_list = self.get_entity().get_component('memory').retrieve(
        query='',  # Don't need a query to retrieve recent memories.
        limit=5,
        scoring_fn=legacy_associative_memory.RetrieveRecent(),
    )
    recent_memories = " ".join(memory.text for memory in recent_memories_list)
    print(f"*****\nDEBUG: Recent memories:\n  {recent_memories}\n*****")
    return recent_memories


class SimpleActing(entity_component.ActingComponent):

  def __init__(self, model: language_model.LanguageModel):
    self._model = model

  def get_action_attempt(
      self,
      contexts,
      action_spec,
  ) -> str:
    # Put context from all components into a string, one component per line.
    context_for_action = "\n".join(
        f"{name}: {context}" for name, context in contexts.items()
    )
    print(f"*****\nDEBUG:\n  context_for_action:\n{context_for_action}\n*****")
    # Ask the LLM to suggest an action attempt.
    call_to_action = action_spec.call_to_action.format(
        name=self.get_entity().name, timedelta='2 minutes')
    sampled_text = self._model.sample_text(
        f"{context_for_action}\n\n{call_to_action}\n",
    )
    return sampled_text


raw_memory = legacy_associative_memory.AssociativeMemoryBank(
    associative_memory.AssociativeMemory(embedder))

# Let's create an agent with the above components.
agent = entity_agent.EntityAgent(
    'Alice',
    act_component=SimpleActing(model),
    context_components={
        'observation': Observe(),
        'recent_memories': RecentMemories(),
        'memory': memory_component.MemoryComponent(raw_memory),
    })

agent.observe("You absolutely hate apples and would never willingly eat them.")
agent.observe("You don't particularly like bananas.")
# Only the next 5 observations will be kept, pushing out critical information!
agent.observe("You are in a room.")
agent.observe("The room has only a table in it.")
agent.observe("On the table there are two fruits: an apple and a banana.")
agent.observe("The apple is shinny red and looks absolutely irresistible!")
agent.observe("The banana is slightly past its prime.")

response3 = agent.act()

print(response3)


class RecentMemoriesImproved(action_spec_ignored.ActionSpecIgnored):

  def __init__(self):
    super().__init__('Recent memories')

  def _make_pre_act_value(self) -> str:
    recent_memories_list = self.get_entity().get_component('memory').retrieve(
        query='',  # Don't need a query to retrieve recent memories.
        limit=5,
        scoring_fn=legacy_associative_memory.RetrieveRecent(),
    )
    recent_memories = " ".join(memory.text for memory in recent_memories_list)
    print(f"*****\nDEBUG: Recent memories:\n  {recent_memories}\n*****")
    return recent_memories

def _recent_memories_str_to_list(recent_memories: str) -> list[str]:
  # Split sentences, strip whitespace and add final period
  return [memory.strip() + '.' for memory in recent_memories.split('.')]


class RelevantMemories(action_spec_ignored.ActionSpecIgnored):

  def __init__(self):
    super().__init__('Relevant memories')

  def _make_pre_act_value(self) -> str:
    recent_memories = self.get_entity().get_component('recent_memories').get_pre_act_value()
    # Each sentence will be used for retrieving new relevant memories.
    recent_memories_list = _recent_memories_str_to_list(recent_memories)
    recent_memories_set = set(recent_memories_list)
    memory = self.get_entity().get_component('memory')
    relevant_memories_list = []
    for recent_memory in recent_memories_list:
      # Retrieve 3 memories that are relevant to the recent memory.
      relevant = memory.retrieve(
          query=recent_memory,
          limit=3,
          scoring_fn=legacy_associative_memory.RetrieveAssociative(add_time=False),
      )
      for mem in relevant:
        # Make sure that we only add memories that are _not_ already in the recent
        # ones.
        if mem.text not in recent_memories_set:
          relevant_memories_list.append(mem.text)
          recent_memories_set.add(mem.text)

    relevant_memories = "\n".join(relevant_memories_list)
    print(f"*****\nDEBUG: Relevant memories:\n{relevant_memories}\n*****")
    return relevant_memories


raw_memory = legacy_associative_memory.AssociativeMemoryBank(
    associative_memory.AssociativeMemory(embedder))

# Let's create an agent with the above components.
agent = entity_agent.EntityAgent(
    'Alice',
    act_component=SimpleActing(model),
    context_components={
        'observation': Observe(),
        'relevant_memories': RelevantMemories(),
        'recent_memories': RecentMemoriesImproved(),
        'memory': memory_component.MemoryComponent(raw_memory),
    })

agent.observe("You absolutely hate apples and would never willingly eat them.")
agent.observe("You don't particularly like bananas.")
# The previous memories will be revtrieved associatively, even though they are
# past the recency limit.
agent.observe("You are in a room.")
agent.observe("The room has only a table in it.")
agent.observe("On the table there are two fruits: an apple and a banana.")
agent.observe("The apple is shinny red and looks absolutely irresistible!")
agent.observe("The banana is slightly past its prime.")

response4 = agent.act()

print(response4)