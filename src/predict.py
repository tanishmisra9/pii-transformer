import spacy
from pathlib import Path
import warnings

# Import the helper functions from evaluate.py
from src.evaluate import load_resources, generate_test_data

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "output" / "model-best"
NUM_EXAMPLES_TO_GENERATE = 10

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

def predict_on_examples():
    """
    Loads the trained model, generates a small, random test set on the fly,
    and prints the model's predictions for each.
    """
    print(f"Loading model from {MODEL_PATH}...")
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please run the training script first (python3 -m src.train)")
        return

    # Load the trained model
    nlp = spacy.load(MODEL_PATH)

    # Generate 10 random test examples on the fly
    print(f"Loading resources and generating {NUM_EXAMPLES_TO_GENERATE} test examples...")
    try:
        resources = load_resources()
        test_data = generate_test_data(*resources, num_samples=NUM_EXAMPLES_TO_GENERATE)
    except FileNotFoundError:
        print("\nError: Could not load resource files (first-names.txt, etc.) from resources/")
        print("Please ensure the 'resources' directory is populated.")
        return

    print("\n" + "="*30)
    print(f"Running predictions on {NUM_EXAMPLES_TO_GENERATE} random examples:")
    print("="*30)

    # Iterate through the generated examples and print entities
    for text, annotations in test_data:
        text = text.strip()
        if not text:
            continue

        doc = nlp(text)
        
        # Get the "actual" entities for comparison
        actual_entities = sorted(list(annotations["entities"]))

        print(f"\nText:      {text}")
        print("Actual:    ", actual_entities)
        
        if doc.ents:
            predicted_entities = sorted([(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents])
            print(f"Predicted: ", predicted_entities)
        else:
            print("Predicted:  []")

if __name__ == "__main__":
    predict_on_examples()