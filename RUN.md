1. Activate Virtual Environment
cd C:\Users\user\1557_VSC\MLF_MVP
.\.venv_ecoflux\Scripts\Activate.ps1


If you're on macOS/Linux:

source .venv_ecoflux/bin/activate

2. Generate Synthetic Dataset

Recreate the dataset used for model training:

cd src
python generate_data.py
cd ..


This produces:

data/energy_synthetic.csv

3. Train Models

Train Linear Regression and MLPRegressor using the dataset:

cd src
python train_models.py
cd ..


This produces:

models/
 ├── ecoflux_linear_regression.pkl
 └── ecoflux_mlp_regressor.pkl

4. Launch the EcoFlux UI
streamlit run app.py


Your browser will automatically open:

http://localhost:8501/
5. Run Unit Tests
pytest tests/


or:

python -m pytest tests/

📦 6. Project Structure Summary
MLF_MVP/
│
├── app.py
├── README.md
├── RUN.md
│
├── data/
│   └── energy_synthetic.csv
│
├── models/
│   ├── ecoflux_linear_regression.pkl
│   └── ecoflux_mlp_regressor.pkl
│
├── notebooks/
│   └── SustainableAI_FinalProjectProtocol.ipynb
│
├── src/
│   ├── __init__.py
│   ├── generate_data.py
│   └── train_models.py
│
└── tests/
    ├── test_prompt_scaling.py
    └── test_data_generation.py