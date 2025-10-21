import spacy
import random
import re
import os
import warnings
from spacy.tokens import DocBin
from random_address import real_random_address
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

warnings.simplefilter("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
RESOURCES_DIR = BASE_DIR / "resources"
DATA_DIR = BASE_DIR / "data"
TEST_RESULTS_DIR = BASE_DIR / "test_results"

FIRST_NAMES_FILE = RESOURCES_DIR / "first-names.txt"
LAST_NAMES_FILE = RESOURCES_DIR / "last-names.txt"
GPE_FILE = RESOURCES_DIR / "gpe.txt"
STRUCTURES_FILE = RESOURCES_DIR / "structures.txt"
ADDRESS_TEMPLATES_FILE = RESOURCES_DIR / "address_sentence_structures.txt"
ADD_TO_NEXT_BATCH_FILE = TEST_RESULTS_DIR / "add_to_next_batch.txt"

PERSON_GPE_SAMPLES = 50000
ADDRESS_SAMPLES = 50000
TRAIN_SPLIT = 0.7

# --- Helper Functions ---

def load_resources():
    with open(FIRST_NAMES_FILE, 'r', encoding="utf-8") as f:
        first_names = f.read().splitlines()
    with open(LAST_NAMES_FILE, 'r', encoding="utf-8") as f:
        last_names = f.read().splitlines()
    with open(GPE_FILE, 'r', encoding="utf-8") as f:
        gpe_list = f.read().splitlines()
    with open(STRUCTURES_FILE, 'r', encoding="utf-8") as f:
        person_structures = f.read().splitlines()
    with open(ADDRESS_TEMPLATES_FILE, 'r', encoding="utf-8") as f:
        address_templates = f.read().splitlines()
    return first_names, last_names, gpe_list, person_structures, address_templates

def generate_person_gpe_data(first_names, last_names, gpe_list, structures, num_samples):
    training_data = []
    for _ in range(num_samples):
        structure = random.choice(structures)
        first_name = random.choice(first_names)
        last_name = random.choice(last_names)
        gpe = random.choice(gpe_list)
        text = structure.replace("FNAME", first_name).replace("LNAME", last_name).replace("GPE", gpe)
        entities = []
        for label, word in [("PERSON", first_name), ("PERSON", last_name), ("GPE", gpe)]:
            for match in re.finditer(r"\b" + re.escape(word) + r"\b", text):
                start, end = match.start(), match.end()
                entities.append([start, end, label])
        training_data.append((text, {"entities": entities}))
    return training_data

def _generate_address_example(templates):
    try:
        address = real_random_address()
        address_line = address['address1']
        template = random.choice(templates)
        sentence = template.replace('HOME_ADDRESS', address_line)
        start = sentence.find(address_line)
        if start == -1:
            return None
        end = start + len(address_line)
        entities = [[start, end, "HOME_ADDRESS"]]
        return (sentence, {"entities": entities})
    except Exception:
        return None

def generate_address_data(templates, num_samples):
    training_data = []
    with ThreadPoolExecutor() as executor:
        results = executor.map(_generate_address_example, [templates] * num_samples)
    return [r for r in results if r is not None]

def load_missed_cases(filepath):
    training_data = []
    if filepath.exists():
        with open(filepath, 'r', encoding="utf-8") as f:
            missed_cases = f.read().splitlines()
        for case in missed_cases:
            try:
                case_text, entity_str = case.split("\t")
                entities = []
                for entity in entity_str.split():
                    start_end, label = entity.split(":")
                    start, end = map(int, start_end.split("-"))
                    entities.append([start, end, label])
                training_data.append((case_text, {"entities": entities}))
            except ValueError:
                continue
    return training_data

def create_spacy_docs(data, nlp):
    docs = []
    for text, annotations in data:
        doc = nlp(text)
        ents = []
        for start, end, label in annotations["entities"]:
            span = doc.char_span(start, end, label=label)
            if span:
                ents.append(span)
        # Remove overlaps
        ents = sorted(ents, key=lambda x: x.start)
        non_overlapping = []
        for span in ents:
            if not any(span.start < e.end and span.end > e.start for e in non_overlapping):
                non_overlapping.append(span)
        doc.ents = non_overlapping
        # Verify data quality
        print(f"\nData quality check:")
        print(f"Total docs created: {len(docs)}")
        print(f"Docs with entities: {sum(1 for doc in docs if doc.ents)}")
        print(f"Sample entities: {docs[0].ents if docs else 'None'}")
        docs.append(doc)
    return docs

def save_docs_to_disk(docs, filename):
    db = DocBin(store_user_data=True)
    for doc in docs:
        db.add(doc)
    db.to_disk(DATA_DIR / filename)
    print(f"Saved {len(docs)} docs to {DATA_DIR / filename}")

# --- Main Execution ---

def main():
    first_names, last_names, gpe_list, person_structures, address_templates = load_resources()
    
    all_training_data = []
    all_training_data.extend(generate_person_gpe_data(first_names, last_names, gpe_list, person_structures, PERSON_GPE_SAMPLES))
    all_training_data.extend(generate_address_data(address_templates, ADDRESS_SAMPLES))
    all_training_data.extend(load_missed_cases(ADD_TO_NEXT_BATCH_FILE))
    
    print(f"Total generated samples: {len(all_training_data)}")
    
    nlp = spacy.blank("en")
    docs = create_spacy_docs(all_training_data, nlp)
    
    # --- Automatically compute labels ---
    labels = set()
    for _, ann in all_training_data:
        for _, _, label in ann["entities"]:
            labels.add(label)
    labels = sorted(labels)
    print(f"Labels for NER: {labels}")
    
    # Save labels to JSON for config initialization
    import json
    with open(DATA_DIR / "ner_labels.json", "w", encoding="utf-8") as f:
        json.dump({l: {} for l in labels}, f)
    
    random.shuffle(docs)
    split_idx = int(TRAIN_SPLIT * len(docs))
    save_docs_to_disk(docs[:split_idx], "train.spacy")
    save_docs_to_disk(docs[split_idx:], "dev.spacy")

if __name__ == "__main__":
    main()
