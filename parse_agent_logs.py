import json
import re
from datetime import datetime
import os
import argparse
from pathlib import Path

def parse_agent_logs(log_content):
    """
    Parse agent logs to create a human-readable format showing prompts,
    responses, components and final decisions.
    
    Args:
        log_content: String containing the raw JSON logs
    
    Returns:
        A formatted string with readable agent interactions
    """
    # Ensure we're working with a string
    if not isinstance(log_content, str):
        log_content = str(log_content)
    # Split the log content into lines and parse each JSON object
    log_entries = []
    for line in log_content.strip().split('\n'):
        try:
            entry = json.loads(line)
            log_entries.append(entry)
        except json.JSONDecodeError:
            continue  # Skip lines that aren't valid JSON
    
    output = []
    current_round = 0
    
    # Group entries by prompt-response-decision sequence
    i = 0
    while i < len(log_entries):
        entry = log_entries[i]
        
        # Handle observations - check if this is a new round
        if entry.get("type") == "observation":
            content = entry.get("content", "")
            timestamp = entry.get("timestamp", "")
            formatted_time = format_timestamp(timestamp)
            
            # Check if this is a new round
            if "Round " in content and "You are in Group" in content:
                round_match = re.search(r"Round (\d+):", content)
                if round_match:
                    new_round = int(round_match.group(1))
                    if new_round != current_round:
                        current_round = new_round
                        output.append(f"\n\n{'='*80}")
                        output.append(f"ROUND {current_round}")
                        output.append(f"{'='*80}\n")
            
            output.append(f"[OBSERVATION {formatted_time}] {content}")
            i += 1
            continue
        
        # Handle prompt-response-decision sequence
        if entry.get("type") == "prompt":
            output.append(f"\n\n{'*'*80}")
            output.append(f"DECISION INTERACTION")
            output.append(f"{'*'*80}\n")
            
            # Process the prompt
            timestamp = entry.get("timestamp", "")
            formatted_time = format_timestamp(timestamp)
            
            output.append(f"=== PROMPT [{formatted_time}] ===")
            component = entry.get("component")
            if component:
                output.append(f"Component: {component}")
            output.append(entry.get("content", ""))
            
            # Look for a corresponding response
            response_entry = None
            concordia_entry = None
            
            j = i + 1
            while j < len(log_entries):
                next_entry = log_entries[j]
                if next_entry.get("type") == "response":
                    response_entry = next_entry
                elif next_entry.get("type") == "concordia_act":
                    concordia_entry = next_entry
                    break
                elif next_entry.get("type") == "prompt":
                    break  # Found the start of the next sequence
                j += 1
            
            # Process the response if found
            if response_entry:
                timestamp = response_entry.get("timestamp", "")
                formatted_time = format_timestamp(timestamp)
                
                output.append(f"\n=== RESPONSE [{formatted_time}] ===")
                component = response_entry.get("component")
                if component:
                    output.append(f"Component: {component}")
                output.append(response_entry.get("content", ""))
            
            # Process the concordia_act if found
            if concordia_entry:
                # Extract final decision
                output.append(f"\n=== FINAL DECISION ===")
                output.append(concordia_entry.get("value", ""))
                
                # Extract components from the prompt field
                prompt_data = concordia_entry.get("prompt", [])
                if prompt_data:
                    components = extract_components(prompt_data)
                    
                    if components:
                        output.append("\n=== COMPONENT BREAKDOWN ===")
                        # Define a specific order for components to display them consistently
                        component_order = [
                            "Observation", 
                            "Recent context", 
                            "Character Assessment", 
                            "Theory of Mind Analysis", 
                            "Theory of Mind Analysis 2",
                            "Emotional State",
                            "Situation Analysis", 
                            "Decision Reflection"
                        ]
                        
                        # First show components in our preferred order
                        for comp_name in component_order:
                            if comp_name in components and components[comp_name]:
                                output.append(f"\n--- {comp_name} ---")
                                output.append(wrap_text(components[comp_name]))
                        
                        # Then show any remaining components that aren't in the preferred order
                        for comp_name, comp_content in components.items():
                            if comp_name not in component_order and comp_content:
                                output.append(f"\n--- {comp_name} ---")
                                output.append(wrap_text(comp_content))
            
            # Skip to after this sequence
            i = j + 1 if concordia_entry else i + 1
        else:
            i += 1  # Skip entries we don't recognize
    
    return "\n".join(output)

def extract_components(prompt_data):
    """Extract component information from the prompt data."""
    components = {}
    
    # Join all the prompt data into a single string to better identify sections
    full_text = "\n".join(str(item) for item in prompt_data if item)
    
    # Look for these specific component patterns
    component_patterns = [
        ("Observation", r'Observation:\s*(.*?)(?=\n[A-Z][\w\s]+:|\Z)'),
        ("Recent context", r'Recent context:\s*(.*?)(?=\n[A-Z][\w\s]+:|\Z)'),
        ("Character Assessment", r'Character Assessment:\s*(.*?)(?=\n[A-Z][\w\s]+:|\Z)'),
        ("Theory of Mind Analysis", r'Theory of Mind Analysis:\s*(.*?)(?=\n[A-Z][\w\s]+:|\Z)'),
        ("Theory of Mind Analysis 2", r'Theory of Mind Analysis 2:\s*(.*?)(?=\n[A-Z][\w\s]+:|\Z)'),
        ("Situation Analysis", r'Situation Analysis:\s*(.*?)(?=\n[A-Z][\w\s]+:|\Z)'),
        ("Decision Reflection", r'Decision Reflection:\s*(.*?)(?=\n[A-Z][\w\s]+:|\Z)'),
        ("Emotional State", r'Emotional State:\s*(.*?)(?=\n[A-Z][\w\s]+:|\Z)')
    ]
    
    # Extract each component
    for component_name, pattern in component_patterns:
        matches = re.search(pattern, full_text, re.DOTALL)
        if matches:
            components[component_name] = matches.group(1).strip()
    
    return components

def format_timestamp(timestamp_str):
    """Format a timestamp string to a more readable format."""
    if not timestamp_str:
        return "Unknown time"
    
    try:
        dt = datetime.fromisoformat(timestamp_str)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return timestamp_str

def extract_decision_type(decision_text):
    """Identify the type of decision from the text."""
    if re.search(r"contribute \$([\d\.]+)", decision_text, re.IGNORECASE):
        return "Contribution"
    elif re.search(r"exclude (Player_\d+|anyone)", decision_text, re.IGNORECASE):
        return "Exclusion"
    elif re.search(r"send (a note|any notes)", decision_text, re.IGNORECASE):
        return "Notes"
    else:
        return "Other"

def wrap_text(text, width=80):
    """Wrap text at word boundaries to improve readability."""
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        # Check if adding this word would exceed the width
        if current_length + len(word) + (1 if current_length > 0 else 0) > width:
            # Add the current line to lines and start a new line
            lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)
        else:
            # Add the word to the current line
            if current_length > 0:
                current_length += 1  # Space before the word
            current_line.append(word)
            current_length += len(word)
    
    # Add the last line if there's anything left
    if current_line:
        lines.append(" ".join(current_line))
    
    return "\n".join(lines)

def process_agent_logs(base_dir):
    """
    Recursively process all agent log files in the given directory structure.
    
    Args:
        base_dir: Base directory containing experiment results
    
    Returns:
        A list of paths to the processed log files
    """
    base_path = Path(base_dir)
    processed_files = []
    
    # Look for agent log directories (they follow the pattern agent_logs_exp_*_*)
    agent_log_dirs = []
    for item in base_path.glob("agent_logs_exp_*_*"):
        if item.is_dir():
            agent_log_dirs.append(item)
    
    if not agent_log_dirs:
        print(f"No agent log directories found in {base_path}")
        return processed_files
    
    print(f"Found {len(agent_log_dirs)} agent log condition directories")
    
    # Process each condition directory
    for condition_dir in agent_log_dirs:
        condition_name = condition_dir.name
        print(f"\nProcessing condition: {condition_name}")
        
        # Find all player directories in this condition
        player_dirs = [d for d in condition_dir.glob("*") if d.is_dir()]
        print(f"Found {len(player_dirs)} player directories")
        
        # Process each player directory
        for player_dir in player_dirs:
            player_name = player_dir.name
            print(f"Processing logs for {player_name} in {condition_name}...")
            
            # Find all log files for this player
            log_files = list(player_dir.glob("*.json")) + list(player_dir.glob("*.jsonl"))
            
            if not log_files:
                print(f"  No log files found for {player_name}")
                continue
            
            print(f"  Found {len(log_files)} log files")
            
            # Process each log file
            for log_file in log_files:
                try:
                    # Read the log file content
                    with open(log_file, 'r', encoding='utf-8') as f:
                        log_content = f.read()
                    
                    # Parse the log content
                    readable_log = parse_agent_logs(log_content)
                    
                    # Create output filename
                    output_file = log_file.parent / f"{log_file.stem}_human_readable.txt"
                    
                    # Save the processed log
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(readable_log)
                    
                    processed_files.append(output_file)
                    print(f"    Processed log saved to {output_file}")
                    
                    # Optionally print the first few lines of output for verification
                    with open(output_file, 'r', encoding='utf-8') as f:
                        first_lines = "".join([f.readline() for _ in range(10)])
                        print(f"    First few lines of output:\n{first_lines}\n    ...")
                    
                except Exception as e:
                    print(f"    Error processing {log_file.name}: {str(e)}")
    
    return processed_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process all agent log files recursively')
    parser.add_argument('--results_dir', help='Base directory containing experiment results')
    
    args = parser.parse_args()
    
    # add prefix to results_dir from the current directory path 
    results_dir = os.path.join(os.path.dirname(__file__), args.results_dir)
    processed_files = process_agent_logs(results_dir)
    
    print(f"\nProcessing complete. {len(processed_files)} files were processed.")

# Command to run:
# python fixed_log_parser.py Multi_Agent/results/03-23-2025_19-39