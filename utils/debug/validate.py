import spacy

def validate_spacy_file(spacy_file):
    try:
        # Load the dataset
        doc_bin = spacy.tokens.DocBin().from_disk(spacy_file)
        docs = list(doc_bin.get_docs(nlp.vocab))
        
        print(f"Validation successful! {len(docs)} documents found in {spacy_file}.")
    except Exception as e:
        print(f"Validation failed: {e}")

# Replace with the actual path to your .spacy file
spacy_file_path = "data/train.spacy"

# Load a blank NLP model
nlp = spacy.blank("en")

# Validate the .spacy file
validate_spacy_file(spacy_file_path)