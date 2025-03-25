#!/usr/bin/env python
# Script to concatenate Python files with clear section labels

import os
import argparse
from pathlib import Path

def concatenate_files(files, output_file, separator_length=80):
    """
    Concatenate multiple files into one with clear section headers.
    
    Args:
        files (list): List of file paths to concatenate
        output_file (str): Path for the output file
        separator_length (int): Length of the separator lines
    """
    with open(output_file, 'w') as outfile:
        for file_path in files:
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} does not exist. Skipping.")
                continue
                
            # Get the filename
            filename = Path(file_path).name
            
            # Create a distinctive header
            separator = "#" * separator_length
            header = f"\n{separator}\n"
            header += f"# FILE: {filename}\n"
            header += f"{separator}\n\n"
            
            # Write the header
            outfile.write(header)
            
            # Write the file contents
            try:
                with open(file_path, 'r') as infile:
                    content = infile.read()
                    outfile.write(content)
                    # Add a newline at the end if needed
                    if not content.endswith('\n'):
                        outfile.write('\n')
            except Exception as e:
                outfile.write(f"# Error reading file: {str(e)}\n")
        
        # Add a final note
        outfile.write(f"\n{separator}\n")
        outfile.write("# END OF CONCATENATED FILES\n")
        outfile.write(f"{separator}\n")

def main():
    parser = argparse.ArgumentParser(description='Concatenate Python files with clear labeling.')
    #parser.add_argument('--files', nargs='+', default=['main.py', 'tpp_scenario.py', 'agent_components.py'],
    parser.add_argument('--files', nargs='+', default=['main_gossip.py', 'gossip_scenario_async.py', 'agent_components_gossip.py', 'custom_classes.py'],
                        help='List of files to concatenate (default: main.py, tpp_scenario.py, agent_components.py)')
    parser.add_argument('--output', type=str, default='combined_code.py',
                        help='Output file path (default: combined_code.py)')
    
    args = parser.parse_args()
    
    print(f"Concatenating files: {', '.join(args.files)}")
    concatenate_files(args.files, args.output)
    print(f"Concatenated output saved to: {args.output}")

if __name__ == "__main__":
    main() 