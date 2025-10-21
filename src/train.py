import subprocess
import time
import sys
from pathlib import Path

# Define project paths
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = BASE_DIR / "config.cfg"
OUTPUT_DIR = BASE_DIR / "output"

def train_model():
    """
    Runs the spaCy training process using the specified config file.
    """
    print("Starting spaCy training...")
    print(f"Config file: {CONFIG_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Command to run: python -m spacy train <config> --output <path>
    command = [
        sys.executable,  # Use the current Python interpreter
        "-m", "spacy",
        "train",
        str(CONFIG_FILE),
        "--output", str(OUTPUT_DIR)
    ]
    
    start_time = time.time()
    
    try:
        # Run the command
        subprocess.run(command, check=True)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        
        print("\n" + "="*30)
        print("Training complete.")
        print(f"Time for completion: {minutes} minutes {seconds} seconds")
        print(f"Models saved to: {OUTPUT_DIR}")
        print("="*30)

    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with exit code {e.returncode}")
    except FileNotFoundError:
        print("\nError: 'spacy' command not found.")
        print("Please ensure spaCy is installed correctly in your environment.")

if __name__ == "__main__":
    train_model()