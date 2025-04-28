# Competitive Pokemon Battle Outcome Predictor

## Overview
Welcome to the Competitive Pokemon Battle Outcome Predictor! 🏋️ This project aims to predict the outcomes of competitive Pokemon battles using machine learning models, offering both **1v1** and **Team vs Team (6v6)** battle prediction features.

Built with:
- **Python** (pandas, scikit-learn, XGBoost)
- **Streamlit** (for UI)
- **Plotly** (for visualizations)

This application provides an interactive way to:
- Select Pokémon based on types.
- Compare Pokémon stats visually.
- Predict 1v1 battle outcomes with win probabilities.
- Predict team battle outcomes (up to 6 Pokémon per side).
- View match histories dynamically.

## Project Motivation
> *"The goal of this project is to predict the outcome of competitive Pokemon battles based on various features such as Pokemon stats, move sets, abilities, and type effectiveness."*

Through structured battle and Pokemon datasets (e.g., Kaggle sources), we leverage machine learning models like **Logistic Regression**, **Random Forest**, and **XGBoost** to predict match outcomes. Feature engineering includes real-world stat calculations and type effectiveness differentials.

---

## Installation and Setup

### 1. Clone the repository
```bash
https://github.com/your_username/Pokemon-Battle-Predictor.git
cd Pokemon-Battle-Predictor
```

### 2. Create a Virtual Environment
It is recommended to use a virtual environment to manage dependencies:
```bash
python -m venv venv
```
Activate the virtual environment:
- **Windows:**
  ```bash
  venv\Scripts\activate
  ```
- **Mac/Linux:**
  ```bash
  source venv/bin/activate
  ```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
*Typical dependencies include:*
- streamlit
- pandas
- scikit-learn
- xgboost
- plotly
- matplotlib
- joblib

### 4. Prepare necessary files
Ensure the following files are present:
- `data/pokemon.csv`
- `data/Pokemon Type Chart.csv`
- `models/model.pkl` (trained XGBoost model)
- `models/scaler.pkl` (preprocessing scaler)

If not, retrain the model using the **EDA_and_modeling.ipynb**.

### 5. Run the app
```bash
streamlit run main.py
```

---

## Usage

1. **1v1 Battle Prediction:**
    - Select two Pokémon from dropdowns (filtered by type if desired).
    - View stat comparison radar charts.
    - Predict battle outcomes and win probabilities.
    - Match history logs for quick reference.

2. **Team vs Team Prediction (6v6):**
    - Select up to 6 Pokémon per team.
    - Randomize teams if preferred.
    - Predict which team has higher chances of winning based on simulation of all pairwise matchups.
    - Team match history tracking.

3. **Clear Match History:**
    - Clear past 1v1 and 6v6 predictions easily from the sidebar.

---

## Core Files Structure

| File | Description |
|:-----|:------------|
| **main.py** | Streamlit app interface for 1v1 and 6v6 predictions. |
| **utils.py** | Data loading, stat calculation, type effectiveness computation, and model-based prediction logic. |
| **EDA_and_modeling.ipynb** | Data exploration, feature engineering, model training and evaluation notebook. |
| **models/model.pkl** | Saved trained model (XGBoost). |
| **models/scaler.pkl** | Saved preprocessing scaler used during model training. |
| **data/pokemon.csv** | Pokemon stat dataset. |
| **data/Pokemon Type Chart.csv** | Type effectiveness reference table. |

---

## Machine Learning Approach

- **Feature Engineering:**
  - Real-world stat computation at Level 50 (IV=31, EV=0).
  - Base Stat Total (BST) excluding Speed.
  - Type Advantage Differential (TypeEff_Diff).
  
- **Model Training:**
  - Models Trained: Logistic Regression, Random Forest, XGBoost.
  - Final Model: **XGBoost**, selected based on best validation accuracy and win probability calibration.

- **Evaluation Metrics:**
  - Accuracy
  - Precision, Recall, F1-Score
  - Confusion Matrix
  - Feature Importance Analysis

---

## Future Improvements
- Integrate **real move sets** and **battle conditions**.
- Explore **Neural Networks (LSTM/Transformer)** to model sequential turns.
- Hyperparameter tuning and ensemble methods.
- Add **leaderboard** feature to track user wins.

---

## Contributors

- **Samuel Castillo**  
  Student, CS4347: Introduction to Machine Learning

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments
- Datasets from Kaggle.
- Streamlit for simple deployment.
- XGBoost for powerful predictive modeling.
- Pokemon by Nintendo, Game Freak, and The Pokemon Company.

> *This project is intended purely for educational purposes.*

