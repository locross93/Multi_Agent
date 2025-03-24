import abc
import threading
import asyncio
from typing_extensions import override

from collections.abc import Collection, Mapping
from typing import Callable
import datetime

from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import logging
from concordia.memory_bank import legacy_associative_memory
from concordia.associative_memory import formative_memories
from concordia.components.agent import memory_component
from concordia.agents import entity_agent_with_logging

from concordia.typing import entity as entity_lib
from concordia.typing import entity_component


class ActionSpecAwareContextComponent(
    entity_component.ContextComponent, metaclass=abc.ABCMeta
):
    """A context component that *does* use the action spec in `pre_act`.

    Like ActionSpecIgnored, but we don't discard the action_spec. Instead,
    `_make_pre_act_value` receives the `action_spec` so you can dynamically
    build your LLM prompt or logic.
    """

    def __init__(self, pre_act_key: str):
        super().__init__()
        self._pre_act_key = pre_act_key
        self._pre_act_value: str | None = None
        self._lock: threading.Lock = threading.Lock()

    @abc.abstractmethod
    def _make_pre_act_value(self, action_spec: entity_lib.ActionSpec) -> str:
        """Subclasses implement their logic here, using the action_spec."""
        raise NotImplementedError()

    def get_pre_act_value(self) -> str:
        """Allows other components to retrieve this pre-act value within PRE_ACT or POST_ACT."""
        phase = self.get_entity().get_phase()
        if phase not in (entity_component.Phase.PRE_ACT, entity_component.Phase.POST_ACT):
            raise ValueError(
                f"get_pre_act_value() is only valid in PRE_ACT or POST_ACT phase, not {phase}."
            )
        with self._lock:
            if self._pre_act_value is None:
                raise ValueError(
                    "No pre_act_value has been set yet. Did you call `pre_act(...)`?"
                )
            return self._pre_act_value

    def get_pre_act_key(self) -> str:
        """Returns the key used as a prefix in the string returned by `pre_act`."""
        return self._pre_act_key

    @override
    def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
        """Compute the pre-act value once per act phase, store it, and return it."""
        with self._lock:
            if self._pre_act_value is None:
                self._pre_act_value = self._make_pre_act_value(action_spec)
            return f"{self._pre_act_key}: {self._pre_act_value}"

    @override
    def update(self) -> None:
        """Clears the cached value after each act cycle so it’s fresh next time."""
        with self._lock:
            self._pre_act_value = None

    def get_named_component_pre_act_value(self, component_name: str) -> str:
        """Returns the pre-act value of a named component of the parent entity."""
        return self.get_entity().get_component(
            component_name, type_=ActionSpecAwareContextComponent).get_pre_act_value()

    @override
    def set_state(self, state: entity_component.ComponentState) -> None:
        # No persistent state by default
        pass

    @override
    def get_state(self) -> entity_component.ComponentState:
        return {}


class QuestionOfRecentMemoriesWithActionSpec(ActionSpecAwareContextComponent):
    """A question-based reflection component, similar to QuestionOfRecentMemories,
    but it uses action_spec.call_to_action as the question.
    """

    def __init__(
        self,
        model: language_model.LanguageModel,
        pre_act_key: str,
        question: str,
        answer_prefix: str,
        add_to_memory: bool,
        memory_tag: str = '',
        memory_component_name: str = memory_component.DEFAULT_MEMORY_COMPONENT_NAME,
        components: Mapping[str, str] = {},
        terminators: Collection[str] = ('\n',),
        clock_now: Callable[[], datetime.datetime] | None = None,
        num_memories_to_retrieve: int = 25,
        logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
    ):
        """
        Args:
          model: The language model to use.
          pre_act_key: Prefix to add to the value of the component when called in `pre_act`.
          answer_prefix: The prefix to add to the answer. (e.g. "{agent_name} would ")
          add_to_memory: Whether to add the answer to the memory.
          memory_tag: The tag to use when adding the answer to memory. (e.g. "[decision reflection]")
          memory_component_name: The name of the memory component from which to retrieve recent memories.
          components: The components to condition the answer on. A mapping of the component name to a label
            to use in the prompt. e.g. {"TheoryOfMind": "Theory of Mind Analysis"}
          terminators: strings that must not be present in the model's response. If
            emitted by the model, the response is truncated before them.
          clock_now: a callback that returns the current datetime, if you want to show time info in the prompt.
          num_memories_to_retrieve: The number of recent memories to retrieve.
          logging_channel: channel to use for debug logging.
        """
        super().__init__(pre_act_key=pre_act_key)
        self._model = model
        self._memory_component_name = memory_component_name
        self._components = dict(components)
        self._clock_now = clock_now
        self._num_memories_to_retrieve = num_memories_to_retrieve
        self._question = question
        self._terminators = terminators
        self._answer_prefix = answer_prefix
        self._add_to_memory = add_to_memory
        self._memory_tag = memory_tag
        self._logging_channel = logging_channel

    def _make_pre_act_value(
        self,
        action_spec,
    ) -> str:
        agent_name = self.get_entity().name

        # 1) Retrieve recent memories
        memory_comp = self.get_entity().get_component(
            self._memory_component_name, 
            type_=memory_component.MemoryComponent
        )
        recency_scorer = legacy_associative_memory.RetrieveRecent(add_time=True)
        recent_mem_objs = memory_comp.retrieve(
            scoring_fn=recency_scorer, limit=self._num_memories_to_retrieve
        )
        mems = '\n'.join(mem.text for mem in recent_mem_objs)

        # 2) Initialize the interactive doc
        prompt = interactive_document.InteractiveDocument(self._model)
        prompt.statement(f"Recent observations of {agent_name}:\n{mems}")

        # 3) Possibly show the current time
        if self._clock_now is not None:
            prompt.statement(f"Current time: {self._clock_now()}.\n")

        # 4) Insert states from other subcomponents
        #    e.g., "TheoryOfMind: (some text), SituationAssessment: (some text), etc."
        component_states = "\n".join(
            f" {prefix}: {self.get_named_component_pre_act_value(key)}"
            for key, prefix in self._components.items()
        )
        prompt.statement(component_states)

        # 5) The big difference: use the action_spec's call_to_action as the question
        question = action_spec.call_to_action
        full_question = self._question.format(agent_name=agent_name, question=question)

        # 6) Let the LLM produce an answer
        answer = prompt.open_question(
            full_question,
            answer_prefix=self._answer_prefix.format(agent_name=agent_name),
            max_tokens=4000,
            terminators=self._terminators,
        )
        # e.g. if answer_prefix = "{agent_name} would ", then we get "Alice would X..."
        final_answer = self._answer_prefix.format(agent_name=agent_name) + answer

        # 7) Optionally store it in memory
        if self._add_to_memory:
            memory_comp.add(f"{self._memory_tag} {final_answer}", metadata={})

        # 8) Log the chain of thought
        log = {
            'Key': self.get_pre_act_key(),
            'Summary': question,
            'State': final_answer,
            'Chain of thought': prompt.view().text().splitlines(),
        }
        if self._clock_now is not None:
            log['Time'] = self._clock_now()
        self._logging_channel(log)

        return final_answer


# Add async capabilities to EntityAgentWithLogging
class AsyncEntityAgent:
    """Async wrapper around EntityAgentWithLogging"""
    
    def __init__(self, agent: entity_agent_with_logging.EntityAgentWithLogging):
        self.agent = agent
        self.name = agent.name
        
    async def act_async(self, action_spec):
        """Async version of the act method"""
        # This would typically use an async version of the LLM call
        # For now, we'll simulate it by running in an executor
        return await asyncio.to_thread(self.agent.act, action_spec)
    
    def observe(self, observation):
        """Pass through observe method"""
        return self.agent.observe(observation)