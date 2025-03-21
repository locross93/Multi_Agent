# agent_logging.py
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

class AgentLogger:
    """Logger to capture all prompts and responses for a specific agent."""
    
    def __init__(self, agent_name: str, log_dir: str = "agent_logs"):
        """Initialize logger for an agent.
        
        Args:
            agent_name: Name of the agent
            log_dir: Directory to store logs
        """
        self.agent_name = agent_name
        self.log_dir = log_dir
        self.interactions = []
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create agent-specific directory
        self.agent_log_dir = os.path.join(log_dir, f"{agent_name}")
        os.makedirs(self.agent_log_dir, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.agent_log_dir, f"{agent_name}_{timestamp}.jsonl")
    
    def log_observation(self, observation: str):
        """Log an observation given to the agent."""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "type": "observation",
            "content": observation
        }
        self._log_interaction(interaction)
    
    def log_prompt(self, prompt: str, component_name: Optional[str] = None):
        """Log a prompt sent to the agent."""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "type": "prompt",
            "component": component_name,
            "content": prompt
        }
        self._log_interaction(interaction)
    
    def log_response(self, response: str, component_name: Optional[str] = None):
        """Log a response from the agent."""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "type": "response",
            "component": component_name,
            "content": response
        }
        self._log_interaction(interaction)
    
    def _log_interaction(self, interaction: Dict[str, Any]):
        """Write an interaction to the log file."""
        self.interactions.append(interaction)
        
        # Write to file immediately for real-time logging
        with open(self.log_file, "a") as f:
            f.write(json.dumps(interaction) + "\n")

def setup_agent_logging(save_dir: str, experiment_id: int, condition: str):
    """Set up agent logging directory structure for an experiment.
    
    Args:
        save_dir: Base directory for saving results
        experiment_id: ID of the experiment
        condition: Experiment condition
        
    Returns:
        Path to the agent logs directory
    """
    # Create experiment-specific log directory
    log_dir = os.path.join(save_dir, f"agent_logs_exp_{experiment_id}_{condition}")
    os.makedirs(log_dir, exist_ok=True)
    
    return log_dir

def create_logger_for_agent(agent_name: str, log_dir: str) -> AgentLogger:
    """Create a logger for a specific agent.
    
    Args:
        agent_name: Name of the agent
        log_dir: Directory to store logs
        
    Returns:
        AgentLogger instance
    """
    return AgentLogger(agent_name=agent_name, log_dir=log_dir)

# Custom agent observation and action wrappers
class LoggingObservationWrapper:
    """Wrapper to log agent observations."""
    
    def __init__(self, agent, logger: AgentLogger):
        self.agent = agent
        self.logger = logger
        
        # Store original observe method
        self.original_observe = agent.observe
        
        # Replace observe method with logging version
        agent.observe = self.observe_with_logging
    
    def observe_with_logging(self, observation: str):
        """Log observation and call original method."""
        self.logger.log_observation(observation)
        return self.original_observe(observation)

class LoggingActionWrapper:
    """Wrapper to log agent actions."""
    
    def __init__(self, agent, logger: AgentLogger):
        self.agent = agent
        self.logger = logger
        
        # Store original act method
        self.original_act = agent.act
        
        # Replace act method with logging version
        agent.act = self.act_with_logging
    
    def act_with_logging(self, action_spec):
        """Log action prompt and response."""
        # Log the prompt/action_spec
        prompt = action_spec.call_to_action
        self.logger.log_prompt(prompt)
        
        # Get the response
        response = self.original_act(action_spec)
        
        # Log the response
        self.logger.log_response(response)
        
        return response

def add_logging_to_agent(agent, logger: AgentLogger):
    """Add logging wrappers to an agent.
    
    Args:
        agent: The agent to add logging to
        logger: AgentLogger instance
        
    Returns:
        Agent with logging wrappers
    """
    LoggingObservationWrapper(agent, logger)
    LoggingActionWrapper(agent, logger)
    return agent