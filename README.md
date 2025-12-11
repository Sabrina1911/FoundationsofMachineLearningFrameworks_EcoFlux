# **EcoFlux: A Prompt-Aware Machine Learning Energy Estimator**
   **Making machine learning greener through transparency**

## Overview

EcoFlux is an educational prototype that demonstrates how **machine learning model design** and **prompt complexity** affect **energy consumption**.  
It allows students to experiment with:

- Number of layers  
- Training time  
- Compute intensity (GFLOPs/hour)  
- Prompt length & linguistic complexity  

EcoFlux predicts the estimated energy consumption (in kWh) and recommends a **lower-energy alternative prompt**.  
The goal is to promote **Green AI**, **energy transparency**, and **responsible prompt engineering**.

## Project Structure

MLF_MVP/
│
├── app.py # Streamlit GUI (final application)
│
├── data/
│ └── energy_synthetic.csv # Auto-generated synthetic dataset
│
├── models/
│ ├── ecoflux_linear_regression.pkl
│ └── ecoflux_mlp_regressor.pkl
│
├── notebooks/
│ └── SustainableAI_FinalProjectProtocol.ipynb
│
└── src/
├── init.py
├── generate_data.py # Synthetic dataset generator
├── train_models.py # Model training + saving

## Features

### **1. ML Energy Estimation**
EcoFlux predicts training energy consumption using:
- Number of layers  
- Training hours  
- FLOPs/hour  
(Linear Regression and MLPRegressor)

### **2. Prompt-Aware Energy Adjustment**
Prompts impact energy via:
- Token count  
- Line density  
- Length-based energy scaling (1.00 → 1.10 multiplier)

### **3. Prompt Optimization**
EcoFlux suggests a lower-energy version of the user’s prompt using:
- Filler phrase removal  
- Phrase compression  
- Template rewrite for Role/Context prompts  

### **4. Side-by-Side Comparison**
The GUI shows:
- Original vs Recommended prompt  
- Token count  
- Complexity score  
- Energy overhead  
- Total predicted energy  

### **5. Clean Interactive Streamlit GUI**
The interface supports:
- Sliders  
- Prompt input  
- Model switching  
- View Prompts expander  
- Sustainability classification (Low / Moderate / High impact)

## How to Run

### 1. Activate virtual environment

cd C:\Users\user\1557_VSC\MLF_MVP
..venv_ecoflux\Scripts\Activate.ps1
pip install -r requiremnets.txt

### 2. Run the GUI

streamlit run app.py
The browser will open automatically.

### Generate synthetic dataset

cd src
python generate_data.py

### Train & save models

python train_models.py

## Model Performance

| Model             | MAE   | MSE   | RMSE  | R²    | Notes                                                  |
|------------------|-------|-------|-------|-------|--------------------------------------------------------|
| Linear Regression | 0.696 | 0.643 | 0.802 | 0.441 | Best performing model; stable & interpretable          |
| MLPRegressor      | 0.725 | 1.081 | 1.040 | 0.060 | Underperforms due to small dataset; slight overfitting |

**Conclusion:**  
Linear Regression is the recommended default model for EcoFlux.

## Testing Summary (All Tests Passed)

| Test Case | Purpose                           | Result |
|----------|-----------------------------------|--------|
| GUI-01   | App launch                        | Pass   |
| GUI-02   | Slider changes update predictions | Pass   |
| GUI-03   | Model switching (Linear ↔ MLP)    | Pass   |
| GUI-04   | Numeric-only changes              | Pass   |
| GUI-05   | Long prompt scaling               | Pass   |
| GUI-06   | Very short prompt                 | Pass   |
| GUI-07   | Empty prompt handling             | Pass   |
| GUI-08   | View Prompts expander             | Pass   |

EcoFlux is fully functional and robust.

## Literature Review Summary

EcoFlux is grounded in findings from **10 influential papers** on Green AI:

- LLMs require massive compute and energy  
- Training one transformer can emit as much CO₂ as multiple cars  
- FLOPs and carbon transparency is essential in ML reporting  
- Prompt length directly affects inference cost  
- Educational tools for ML energy awareness are lacking  

EcoFlux fills these gaps by providing a **hands-on, transparent simulation tool**.

## Why EcoFlux Matters

EcoFlux helps students understand:

- The hidden cost of “bigger models”  
- How design decisions impact sustainability  
- Why prompt engineering matters for efficiency  
- How to reduce unnecessary computation  

It supports the shift toward **responsible, environmentally conscious AI practices**.

## Future Enhancements

EcoFlux can be upgraded with:

- Real tokenization (BPE/WordPiece)  
- Carbon-intensity lookup by region  
- Random Forest / Gradient Boosting regressors  
- Embedding-based semantic prompt rewriting  
- Deployment as a web API  


