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
    
    def log_component_output(self, component_name: str, output: str):
        """Log output from an internal component."""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "type": "component_output",
            "component": component_name,
            "content": output
        }
        self._log_interaction(interaction)
    
    def _log_interaction(self, interaction: Dict[str, Any]):
        """Write an interaction to the log file."""
        self.interactions.append(interaction)
        
        # Write to file immediately for real-time logging
        with open(self.log_file, "a") as f:
            f.write(json.dumps(interaction) + "\n")

    def log_concordia_act(self, concordia_log: Dict[str, Any]):
        """Log a complete Concordia act component log."""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "type": "concordia_act",
            "key": concordia_log.get('Key', 'Unknown'),
            "value": concordia_log.get('Value', ''),
            "prompt": concordia_log.get('Prompt', [])
        }
        self._log_interaction(interaction)

# Keep the existing wrappers
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
        
        # Capture the Concordia native log if available
        try:
            # Get the last log entry which contains the full prompt and response
            log_entry = self.agent.get_last_log()
            if log_entry and 'ActComponent' in log_entry:
                concordia_log = log_entry['ActComponent']
                
                # Add a new method to AgentLogger to log this
                self.logger.log_concordia_act(concordia_log)
                
                # Debug print
                print(f"Captured Concordia log for {self.agent.name}")
        except Exception as e:
            print(f"Warning: Could not capture Concordia log: {e}")
        
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

def setup_measurement_logging(measurements, agent_loggers_dict):
    """Set up logging from measurement channels to agent loggers."""
    
    # Debug: Print available channels
    print(f"Available channels: {dir(measurements)}")
    
    # Define a handler that routes component outputs to the appropriate agent logger
    def log_component_output(data):
        # Debug print
        print(f"Got component output: {data}")
        
        # Check if the data contains agent information
        if 'agent' in data and 'text' in data:
            agent_name = data['agent']
            component_name = data.get('component', 'Unknown')
            text = data['text']
            
            # Route to the appropriate agent logger
            if agent_name in agent_loggers_dict:
                agent_loggers_dict[agent_name].log_component_output(component_name, text)
    
    # Subscribe to component channels
    for channel_name in [
        'PersonalityReflection',
        'SituationAssessment',
        'TheoryOfMind',
        'ContributionDecision',
        'GossipDecision',
        'OstracismDecision',
        'ActComponent'
    ]:
        try:
            channel = measurements.get_channel(channel_name)
            print(f"Successfully got channel: {channel_name}")
            channel.subscribe(log_component_output)
            print(f"Subscribed to channel: {channel_name}")
        except Exception as e:
            print(f"Warning: Could not subscribe to channel {channel_name}: {e}")


def add_logging_to_experiment(
    agents, 
    measurements,
    log_dir
):
    """Add comprehensive logging to all agents in an experiment.
    
    Args:
        agents: List of agents
        measurements: Concordia measurements object
        log_dir: Directory to store logs
        
    Returns:
        Dictionary mapping agent names to loggers
    """
    # Create loggers for all agents
    agent_loggers_dict = {}
    
    for agent in agents:
        logger = create_logger_for_agent(agent.name, log_dir)
        add_logging_to_agent(agent, logger)
        agent_loggers_dict[agent.name] = logger
    
    # Set up measurement logging
    setup_measurement_logging(measurements, agent_loggers_dict)
    
    return agent_loggers_dict


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