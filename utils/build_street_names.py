import csv
import random
import re
import time
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from random_address import real_random_address

# --- Configuration ---

# Define project paths assuming this script is in `utils/`
BASE_DIR = Path(__file__).resolve().parent.parent
RESOURCES_DIR = BASE_DIR / "resources"
UTILS_DIR = BASE_DIR / "utils"

# Source file (assumed to be in `utils/` as it's a build source)
CSV_SOURCE_FILE = UTILS_DIR / "street_names.csv" 

# Target file
FINAL_STREET_NAMES_FILE = RESOURCES_DIR / "street_names.txt"

# Number of random names to generate
NUM_TO_GENERATE = 1_000_000 

# --- Helper Functions ---

def add_number_to_street(street_name):
    """
    Applies custom logic to prepend a number to a street name.
    Based on the logic from csv_to_txt.py.
    
    Args:
        street_name (str): The street name.

    Returns:
        str: The street name with a prepended number.
    """
    parts = street_name.split()
    if not parts:
        return street_name  # Return original if empty

    first_part = parts[0]
    suffix = parts[-1]

    # 1. Check for ordinal numbers (e.g., "1St", "42Nd")
    if re.match(r"^\d+(St|Nd|Rd|Th)$", first_part, re.IGNORECASE):
        street_name = f"0 {street_name}"
    # 2. Check if first part isn't a number and suffix is short
    elif not first_part[0].isdigit() and len(suffix) < 2:
        street_name = f"0 {street_name}"
    # 3. Else, add a random number
    else:
        street_name = f"{random.randint(0, 6)} {street_name}"

    return street_name

def process_csv_names(csv_path):
    """
    Reads street names from a CSV, processes them, and returns a set.
    Based on logic from csv_to_txt.py.

    Args:
        csv_path (Path): Path to the source CSV file.

    Returns:
        set: A set of processed street names.
    """
    processed_names = set()
    if not csv_path.exists():
        print(f"Warning: CSV source file not found at {csv_path}")
        return processed_names

    print(f"Processing names from {csv_path}...")
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        try:
            next(reader)  # Skip header
        except StopIteration:
            return processed_names # File is empty

        for row in reader:
            if not row:
                continue
            
            street_name = row[0].title()
            
            if street_name.lower() != "unnamed":
                # Capitalize suffixes
                parts = street_name.split()
                if parts and parts[-1].upper() in ["AVE", "ST", "RD", "BLVD", "LN", "DR"]:
                    parts[-1] = parts[-1].capitalize()
                    street_name = " ".join(parts)
                
                street_name = add_number_to_street(street_name)
                processed_names.add(street_name)
    
    print(f"Found {len(processed_names)} unique names from CSV.")
    return processed_names

def _get_random_street_name(_):
    """Helper for multithreading. Fetches one random street name."""
    try:
        address = real_random_address()
        # Extract just the street line (e.g., "123 Main St")
        return address.get('address1', '').split(',')[0]
    except Exception:
        return None  # Handle API errors safely

def generate_random_names(n=1_000_000):
    """
    Generates a large number of random street names using multithreading.
    Based on logic from gen_mil_add.py.

    Args:
        n (int): The number of names to generate.

    Returns:
        set: A set of generated street names.
    """
    print(f"Generating {n} random street names (multithreaded)...")
    start_time = time.time()
    
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(_get_random_street_name, range(n)))
    
    # Filter out None results from errors and empty strings
    generated_names = {name for name in results if name}
    
    elapsed = time.time() - start_time
    print(f"Generated {len(generated_names)} unique random names in {elapsed:.2f} sec")
    return generated_names

def load_existing_names(filepath):
    """
    Loads names from an existing text file, if it exists.

    Args:
        filepath (Path): The path to the text file.

    Returns:
        set: A set of names from the file.
    """
    if not filepath.exists():
        return set()
    
    print(f"Loading existing names from {filepath}...")
    with open(filepath, "r", encoding="utf-8") as f:
        names = set(f.read().splitlines())
    print(f"Loaded {len(names)} existing names.")
    return names

# --- Main Execution ---

def main():
    """
    Runs the full pipeline to build the street_names.txt resource:
    1. Loads existing names from the target file.
    2. Processes new names from the source CSV.
    3. Generates a large set of random names.
    4. Merges all, deduplicates, sorts, and overwrites the target file.
    """
    print("Starting build process for street_names.txt...")

    # 1. Load existing names
    all_names = load_existing_names(FINAL_STREET_NAMES_FILE)

    # 2. Process names from CSV
    csv_names = process_csv_names(CSV_SOURCE_FILE)
    all_names.update(csv_names)

    # 3. Generate random names
    random_names = generate_random_names(NUM_TO_GENERATE)
    all_names.update(random_names)

    # 4. Sort and save
    print(f"\nTotal unique names compiled: {len(all_names)}")
    sorted_names = sorted(list(all_names))

    # Ensure the resources directory exists
    RESOURCES_DIR.mkdir(exist_ok=True)
    
    with open(FINAL_STREET_NAMES_FILE, "w", encoding="utf-8") as f_out:
        f_out.write("\n".join(sorted_names))
    
    print(f"Successfully saved all names to {FINAL_STREET_NAMES_FILE}")

if __name__ == "__main__":
    main()