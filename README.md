# Language and AI – Interim Assignment (Group 24)

This repository contains the code for the interim assignment of the Language and AI course.  
We study nationality prediction from Reddit posts and analyze the effect of data pollution when additional information from political leaning prediction is added.

## Structure

- `src/data/`  
  Data loading and preprocessing code.  
  The original datasets (`nationality.csv`, `political_leaning.csv`) are not included.

- `src/models/`  
  Training scripts for all models (nationality - polluted and clean, political leaning - clean, and combined models - polluted and clean).
  Evaluation script used to compare model performance.
  
- `outputs/models/`  
  Locally saved trained models (not tracked by git).

- `outputs/data/`  
  Generated datasets (nationality data augmented with predicted political leaning).

  - `outputs/results/`
    Results of the evaluation script

## Models

We train the following models:
- **N1**: Nationality prediction from post text  
- **N1-clean**: Same as N1, with country names masked  
- **P**: Political leaning prediction (cleaned text)  
- **N2**: Nationality prediction using text and predicted political leaning  
- **N2-clean**: Same as N2, with country names masked  

All models use TF–IDF features and logistic regression.

## Running the code

To reproduce the experiments:
1. Place the original CSV files in `src/data/`
2. Install the required Python packages:
   - `pandas`
   - `scikit-learn`
   - `joblib`
3. Run the training scripts in `src/models/`
4. Use the evaluation scripts in `src/eval/` to obtain the reported results

Trained models and generated data are intentionally not included in the repository.
