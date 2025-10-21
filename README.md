# Transformer-based PII Extraction

This project is a robust PII (Personally Identifiable Information) identification system built using Python and a spaCy Transformer model.

## Description

  * Developed a Python-based pipeline using a spaCy Transformer model to build a robust PII identification system.
  * Created a custom data generation script to create over 1 million records by scraping public datasets and using synthetic data generation to ensure data privacy and ethical standards.
  * Analyzed model performance and improved accuracy from 22% to over 90% through iterative, intentional adjustments to the model's configuration.

### Technologies Used

  * Python
  * spaCy
  * `spacy-transformers`
  * `random_address`

-----

## Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/tanishmisra9/pit-transformer.git
    ```

2.  **Navigate to the project directory:**

    ```bash
    cd pit-transformer
    ```

3.  **Install the required dependencies:**
    This project requires two main packages. Install them using pip:

    ```bash
    # Install spaCy and the transformer pipeline components
    pip install -U "spacy[transformers]"

    # Install the library for generating synthetic addresses
    pip install random_address
    ```

-----

## Usage Workflow

The project is designed to be run as a four-step pipeline from your terminal.

### 1\. Generate Training Data

This script reads from the `resources/` folder, generates thousands of training samples (for `PERSON`, `GPE`, and `HOME_ADDRESS`), and saves the `train.spacy` and `dev.spacy` files to the `data/` directory.

```bash
python3 -m src.data_generator
```

### 2\. Train the Model

This script runs the main spaCy training process using the `config.cfg` file. It loads the data from the `data/` directory and saves the final trained models (`model-best` and `model-last`) to the `output/` directory.

```bash
python3 -m src.train
```

### 3\. Evaluate the Model

After training, this script loads the `model-best` from the `output/` directory and runs it against a new, randomly generated test set. It prints the final accuracy and saves a detailed log to the `test_results/` folder.

```bash
python3 -m src.evaluate
```

### 4\. Run a Quick Prediction

This script provides a simple way to test the trained model. It loads `model-best` and **generates 10 random test examples on the fly**, printing the actual vs. predicted entities for each.

```bash
python3 -m src.predict
```

-----

## Project Structure

```
pit-transformer/
│
├── .gitignore
├── README.md
├── config.cfg                  # The main spaCy training configuration
├── HYPERPARAMETER_TUNING.txt   # Notes on tuning the model
│
├── src/                        # All main Python source code
│   ├── data_generator.py       # Step 1: Creates .spacy training files
│   ├── train.py                # Step 2: Runs the spaCy training
│   ├── evaluate.py             # Step 3: Tests the trained model
│   └── predict.py              # Step 4: Runs inference on random samples
│
├── resources/                  # Raw text files for data generation
│
├── data/                       # (Generated) Directory for .spacy files
│
├── output/                     # (Generated) Directory for trained models
│
└── test_results/               # (Generated) Directory for evaluation logs
```
