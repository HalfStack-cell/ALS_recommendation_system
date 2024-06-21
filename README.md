# ALS Recommendation System

This repository contains the implementation of an Alternating Least Squares (ALS) recommendation system. The project demonstrates how to build, train, evaluate, and fine-tune an ALS model to predict course recommendations based on user ratings.

## Project Structure

- **data/**: Contains the synthetic data used for the model.
  - `synthetic_data.csv`: The synthetic dataset used for demonstration purposes.
- **models/**: Directory where the trained ALS model is saved.
- **scripts/**: Contains the scripts to run the different stages of the project.
  - `als_recommendation.py`: Script to train the ALS model.
  - `als_inference.py`: Script to generate recommendations using the trained ALS model.
  - `als_evaluation.py`: Script to evaluate the ALS model and perform hyperparameter tuning.
  - `inspect_users_id.py`: Script to inspect user IDs in the dataset.
  - `data_preparation.py`: Script for preparing and cleaning the data.
- **requirements.txt**: List of Python dependencies needed to run the project.
- **README.md**: Project documentation.

## Installation

To set up the project on your local machine, follow these steps:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/HalfStack-cell/ALS_recommendation_system.git
    cd ALS_recommendation_system
    ```

2. **Create a virtual environment and activate it**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### 1. Data Preparation


Ensure that the synthetic data is available in the `data/` directory. If not, generate it using the `data_preparation.py` script:
```sh
python scripts/data_preparation.py


##Training the model
python scripts/als_recommendation.py

##Making predictions
python scripts/als_inference.py

##Model Evaluation and hyperparameter Tuning
python scripts/als_evaluation.py


##This script will output the root-mean-square error (RMSE) and the best parameters found during tuning.
