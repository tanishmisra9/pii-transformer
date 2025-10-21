import spacy

# Load a pretrained model instead of blank model
# nlp = spacy.load("output/model-best")

nlp = spacy.load("en_core_web_trf")

ner = nlp.get_pipe("ner")

# Print all entity labels the model recognizes
print("Entities recognized by the model:")
print(ner.labels)