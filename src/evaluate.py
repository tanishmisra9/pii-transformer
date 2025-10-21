import spacy
import random
import os
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
RESOURCES_DIR = BASE_DIR / "resources"
MODEL_PATH = BASE_DIR / "output" / "model-best"
TEST_RESULTS_DIR = BASE_DIR / "test_results"

# Resource files
FIRST_NAMES_FILE = RESOURCES_DIR / "first-names.txt"
LAST_NAMES_FILE = RESOURCES_DIR / "last-names.txt"
GPE_FILE = RESOURCES_DIR / "gpe.txt"
STRUCTURES_FILE = RESOURCES_DIR / "structures.txt"

# Evaluation settings
NUM_TEST_SAMPLES = 500

# --- Helper Functions ---

def load_resources():
    """Loads resources needed for generating test data."""
    with open(FIRST_NAMES_FILE, 'r') as f:
        first_names = f.read().splitlines()
    with open(LAST_NAMES_FILE, 'r') as f:
        last_names = f.read().splitlines()
    with open(GPE_FILE, 'r') as f:
        gpe_list = f.read().splitlines()
    with open(STRUCTURES_FILE, 'r') as f:
        sentence_structures = f.read().splitlines()
    return first_names, last_names, gpe_list, sentence_structures

def generate_test_data(first_names, last_names, gpe_list, sentence_structures, num_samples):
    """
    Generates a new, unseen test dataset on the fly.
    
    Args:
        first_names (list): List of first names.
        last_names (list): List of last names.
        gpe_list (list): List of GPE names.
        structures (list): List of sentence templates.
        num_samples (int): Number of test samples to generate.

    Returns:
        list: A list of (text, {"entities": [...]}) tuples.
    """
    test_data = []
    
    # Select all samples in one go for efficiency
    first_names_sample = random.choices(first_names, k=num_samples)
    last_names_sample = random.choices(last_names, k=num_samples)
    gpe_sample = random.choices(gpe_list, k=num_samples)
    structures_sample = random.choices(sentence_structures, k=num_samples)

    for i in range(num_samples):
        structure = structures_sample[i]
        first_name = first_names_sample[i]
        last_name = last_names_sample[i]
        gpe = gpe_sample[i]

        text = structure.replace("FNAME", first_name).replace("LNAME", last_name).replace("GPE", gpe)
        
        entities = []
        for label, word in [("PERSON", first_name), ("PERSON", last_name), ("GPE", gpe)]:
            for match in re.finditer(r"\b" + re.escape(word) + r"\b", text):
                entities.append((match.start(), match.end(), label))

        test_data.append((text, {"entities": entities}))
    
    return test_data

def test_model(nlp, test_data):
    """
    Evaluates the model on the test data using batch processing.
    
    Args:
        nlp (spacy.Language): The trained spaCy model.
        test_data (list): List of (text, entities) tuples.

    Returns:
        tuple: (accuracy, incorrect_cases, results_log)
    """
    texts = [text for text, _ in test_data]
    docs = list(nlp.pipe(texts))  # Batch processing

    total = len(test_data)
    correct = 0
    incorrect_cases = []
    results_log = []

    for doc, (text, annotations) in zip(docs, test_data):
        predicted_entities = {(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents}
        actual_entities = set(annotations["entities"])

        if predicted_entities == actual_entities:
            correct += 1
        else:
            incorrect_cases.append((text, annotations["entities"]))

        # Log each test case in detail
        results_log.append(f"Sentence: {text}\n")
        results_log.append(f"Predicted: {sorted(list(predicted_entities))}\n")
        results_log.append(f"Actual: {sorted(list(actual_entities))}\n")
        results_log.append("=" * 80 + "\n")

    accuracy = (correct / total) * 100
    return accuracy, incorrect_cases, results_log

def save_results(accuracy, incorrect_cases, results_log):
    """
    Saves all evaluation outputs to the test_results directory.
    
    Args:
        accuracy (float): Model accuracy percentage.
        incorrect_cases (list): List of (text, entities) for misclassified samples.
        results_log (list): Detailed string log of all predictions.
    """
    print("Saving evaluation results...")
    TEST_RESULTS_DIR.mkdir(exist_ok=True)

    # 1. Save misclassified cases for re-training
    with open(TEST_RESULTS_DIR / "add_to_next_batch.txt", "w") as f:
        for text, entities in incorrect_cases:
            entity_str = " ".join([f"{start}-{end}:{label}" for start, end, label in entities])
            f.write(f"{text}\t{entity_str}\n")
    print(f"Saved {len(incorrect_cases)} misclassified cases to test_results/add_to_next_batch.txt")

    # 2. Save accuracy to a log file
    with open(TEST_RESULTS_DIR / "accuracy_log.txt", "a") as f:
        f.write(f"{accuracy:.2f}%\n")
    print(f"Accuracy appended to test_results/accuracy_log.txt")

    # 3. Save detailed output for review
    with open(TEST_RESULTS_DIR / "testing_output.txt", "w") as f:
        f.writelines(results_log)
    print("Detailed testing results saved to test_results/testing_output.txt")

# --- Main Execution ---

def main():
    """
    Main function to load the model, generate test data, and run evaluation.
    """
    print(f"Loading model from {MODEL_PATH}...")
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please run the training script first (python -m src.train)")
        return

    nlp = spacy.load(MODEL_PATH)
    
    print("Loading resources and generating test data...")
    resources = load_resources()
    test_data = generate_test_data(*resources, num_samples=NUM_TEST_SAMPLES)
    
    print(f"Running evaluation on {len(test_data)} test samples...")
    accuracy, incorrect_cases, results_log = test_model(nlp, test_data)
    
    print("\n" + "="*30)
    print(f"Model Accuracy: {accuracy:.2f}%")
    print("="*30 + "\n")
    
    save_results(accuracy, incorrect_cases, results_log)

if __name__ == "__main__":
    main()