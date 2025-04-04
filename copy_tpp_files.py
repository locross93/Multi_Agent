import os
import shutil
import re
from pathlib import Path

def copy_and_renumber_files():
    # Source directories
    source_dirs = [
        "results/tpp/03-28-2025_17-16",
        "results/tpp/03-28-2025_17-17",
        "results/tpp/03-28-2025_17-18",
        "results/tpp/03-28-2025_17-19",
    ]
    
    # Destination directory
    dest_dir = "results/tpp/03-28-2025_17-20"
    
    # Make sure destination directory exists
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    
    # Regex pattern to extract experiment number
    pattern = re.compile(r'tpp_exp_(\d+)_public\.json')

    delta = 20
    
    # Process each source directory
    for source_dir in source_dirs:
        print(f"Processing directory: {source_dir}")
        
        # Get all JSON files in the directory
        json_files = list(Path(source_dir).glob("tpp_exp_*_public.json"))
        
        for file_path in json_files:
            filename = file_path.name
            match = pattern.match(filename)
            
            if match:
                exp_number = int(match.group(1))
                
                # Only process files with experiment numbers 0-39
                if 0 <= exp_number <= 39:
                    # Create new filename with number + 20
                    new_exp_number = exp_number + delta
                    new_filename = f"tpp_exp_{new_exp_number}_public.json"
                    dest_path = Path(dest_dir) / new_filename
                    
                    # Copy the file
                    shutil.copy2(file_path, dest_path)
                    print(f"Copied: {filename} → {new_filename}")

        delta += 20
    
    print("File copying completed!")

if __name__ == "__main__":
    copy_and_renumber_files() 