import numpy as np
import pandas as pd
from pathlib import Path

import streamlit as st
import joblib
import matplotlib.pyplot as plt  # NEW: for baseline energy curve plot
import re  # NEW: for case-insensitive, pattern-based simplification

# --------------------------------------------------------
# EcoFlux: A Prompt-Aware Machine Learning Energy Estimator
# Streamlit GUI
#
# This version:
# - uses the trained regression models (Linear / MLP)
# - adds a prompt-complexity feature at inference time
# - recommends a shorter, lower-energy prompt
# --------------------------------------------------------

# 1. Page configuration
st.set_page_config(
    page_title="EcoFlux – Energy Estimator",
    page_icon="🌿",
    layout="wide",
)

# --------------------------------------------------------
# 2. Helper functions
# --------------------------------------------------------

@st.cache_resource
def load_models(models_dir: Path):
    """
    Load trained models from the given directory.

    Returns
    -------
    lin_model : sklearn LinearRegression
    mlp_bundle : dict with keys {"scaler", "model"}
    """
    lin_path = models_dir / "ecoflux_linear_regression.pkl"
    mlp_path = models_dir / "ecoflux_mlp_regressor.pkl"

    lin_model = joblib.load(lin_path)
    mlp_bundle = joblib.load(mlp_path)

    return lin_model, mlp_bundle

@st.cache_data
def load_energy_data(csv_path: Path) -> pd.DataFrame:
    """
    Load the synthetic energy dataset used for training the regression model.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    return pd.read_csv(csv_path)


def classify_sustainability(energy_kwh: float, baseline_kwh: float = 2.0):
    """
    Map a numeric energy value to a simple sustainability label.
    Baseline is used only for the delta, not for the thresholds.
    """
    if energy_kwh < 1.5:
        label = "🟢 Low impact"
        color = "green"
    elif energy_kwh < 3.0:
        label = "🟡 Moderate impact"
        color = "orange"
    else:
        label = "🔴 High impact"
        color = "red"

    delta = energy_kwh - baseline_kwh
    return label, color, delta

def load_iso_model(models_dir: Path):
    """
    Load the pre-trained IsolationForest model for anomaly detection.
    The model should be trained offline and saved as a .pkl file.
    """
    iso_path = models_dir / "ecoflux_iso_forest.pkl"
    iso_model = joblib.load(iso_path)
    return iso_model


def classify_anomaly(
    num_layers: float,
    training_hours: float,
    flops_per_hour: float,
    base_energy: float,
    iso_model,
    score_threshold_warn: float = -0.4,
):
    """
    Use IsolationForest to classify whether the current configuration
    is normal, unusual, or anomalous.

    Parameters
    ----------
    num_layers : float
        Model depth (layers).
    training_hours : float
        Training duration in hours.
    flops_per_hour : float
        Compute intensity in GFLOPs/hour.
    base_energy : float
        Base energy prediction (kWh) from the regression model.
        Include this only if the IsolationForest was trained with energy.
    iso_model :
        Fitted IsolationForest instance loaded from disk.
    score_threshold_warn : float
        Threshold on the anomaly score to flag "unusual" cases.

    Returns
    -------
    status_text : str
        Human-readable anomaly status.
    status_color : str
        Color name for UI (e.g., 'green', 'orange', 'red').
    score : float
        Raw anomaly score (higher = more normal, lower = more anomalous).
    """
    # IMPORTANT: The feature order and dimensionality must match
    # how you trained the IsolationForest offline.
    # Example: 4 features [num_layers, training_hours, flops_per_hour, base_energy]
    X_input_iso = np.array([[num_layers, training_hours, flops_per_hour, base_energy]])

    pred_label = iso_model.predict(X_input_iso)[0]      # 1 = normal, -1 = anomaly
    score = iso_model.score_samples(X_input_iso)[0]     # higher = more normal

    if pred_label == -1:
        status_text = "🔴 Anomalous configuration (out-of-distribution)"
        status_color = "red"
    else:
        # "Normal but unusual" vs "normal and typical"
        if score < score_threshold_warn:
            status_text = "🟡 Unusual but not critical"
            status_color = "orange"
        else:
            status_text = "🟢 Normal configuration"
            status_color = "green"

    return status_text, status_color, score

def compute_prompt_features(prompt: str):
    """
    Compute simple prompt-complexity features from raw text.

    Parameters
    ----------
    prompt : str
        User-entered prompt.

    Returns
    -------
    token_count : int
        Number of whitespace-separated tokens.
    line_count : int
        Number of non-empty lines.
    complexity_score : float
        Normalised score (roughly 0–1) based on length and density.
    """
    # Split on whitespace for a simple token approximation
    word_tokens = prompt.split()
    word_token_count = len(word_tokens)

    # newline tokens: count each '\n' as one extra token
    newline_count = prompt.count("\n")

    # Final token count
    token_count = word_token_count + newline_count

    # Count non-empty lines
    lines = [ln for ln in prompt.splitlines() if ln.strip()]
    line_count = max(1, len(lines))  # avoid divide-by-zero

    # Average tokens per non-empty line
    avg_tokens_per_line = token_count / line_count

    # Normalise into a 0–1 band (cap at typical classroom prompt sizes)
    # 0  tokens -> 0.0, 200+ tokens -> ~1.0
    length_component = min(1.0, token_count / 200.0)
    complexity_score = length_component

    # density_component = min(1.0, avg_tokens_per_line / 40.0)

    # Simple average of the two components
    # complexity_score = 0.5 * (length_component + density_component)

    return token_count, line_count, complexity_score


def suggest_simpler_prompt(prompt: str, max_tokens: int = 80):
    """
    Create a shorter version of the prompt using simple heuristics.

    Cases:
    - Empty / whitespace-only: return as-is
    - Very short prompts (<= 10 tokens): return as-is
    - Long 'Role: / Context: / Expectation: / Final task:' prompts:
        -> return a concise, hand-crafted template preserving intent
    - Other prompts:
        -> remove filler phrases + clip to max_tokens with ellipsis
    """
    # Normalise whitespace
    cleaned = " ".join(prompt.strip().split())

    if not cleaned:
        return cleaned

    # Tokenise once for basic length decisions
    tokens = cleaned.split()
    token_count = len(tokens)

    # --- 1) Very short prompts: don't touch them ---
    if token_count <= 10:
        return cleaned

    # --- 2) Special handling for long Role/Context/Expectation style prompts ---
    has_role = "Role:" in cleaned
    has_context = "Context:" in cleaned

    if has_role and has_context and token_count > max_tokens:
        # Extract the first sentence from the Role: section, if possible
        try:
            role_start = cleaned.index("Role:")
            # From "Role:" to the end
            role_fragment = cleaned[role_start:]
            # Cut at the first period
            dot_idx = role_fragment.find(".")
            if dot_idx != -1:
                role_sentence = role_fragment[: dot_idx + 1]
            else:
                role_sentence = role_fragment
        except ValueError:
            # Fallback: if indexing fails, just keep a generic role sentence
            role_sentence = "Role: You are a sustainability-focused AI assistant."

        simplified = (
            f"{role_sentence} "
            "Context: Summarise how large language models affect energy use, carbon footprint, "
            "and sustainability. "
            "Expectation: Explain trade-offs between model performance and compute cost, and "
            "mention renewable energy, efficient hardware, and responsible AI practices. "
            "Final task: End with 3–4 bullet-point takeaways linked to modern sustainability goals."
        )

        # Clean up any extra spaces
        return " ".join(simplified.split())

    # --- 3) Generic simplification for other (non-template) prompts ---
    # Remove / compress verbose phrases (case-insensitive)
    filler_phrases = [
        "please write in detail about",
        "please provide a detailed explanation of",
        "in as much detail as possible",
        "you are an expert",
        "act as an expert",
        "suitable for beginners",
        "suitable for beginner readers",
        "keep the explanation simple and clear",
        "keep it simple and clear",
        "in simple and clear language",
        "in one sentence",
        "in a single sentence",
        "highly detailed",
        "deeply detailed",
        "multi-paragraph explanation",
        "highly structured",
    ]

    for phrase in filler_phrases:
        cleaned = re.sub(re.escape(phrase), "", cleaned, flags=re.IGNORECASE)

    # Special compression examples
    cleaned = re.sub(
        r"\btwo examples suitable for beginners\b",
        "two simple examples",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\bprovide a highly detailed, multi-paragraph explanation\b",
        "provide a clear explanation",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Normalise whitespace again after removals
    cleaned = " ".join(cleaned.split())
    tokens = cleaned.split()
    token_count = len(tokens)

    # If still short enough, return as-is
    if token_count <= max_tokens:
        return cleaned

    # Otherwise clip to max_tokens and add ellipsis
    shortened = " ".join(tokens[:max_tokens]) + " ..."
    return shortened

def prompt_scaling_factor(token_count: int) -> float:
    """
    High granularity scaling: 10 buckets
    """
    if token_count <= 10:
        return 1.00
    elif token_count <= 20:
        return 1.02
    elif token_count <= 40:
        return 1.04
    elif token_count <= 60:
        return 1.06
    elif token_count <= 80:
        return 1.08
    elif token_count <= 100:
        return 1.10
    elif token_count <= 140:
        return 1.12
    elif token_count <= 180:
        return 1.14
    elif token_count <= 240:
        return 1.17
    else:
        return 1.20
    
def prompt_overhead_kwh(token_count: int, complexity: float) -> float:
    """
    Estimate additional energy overhead (in kWh) due to prompt length/complexity.

    Parameters
    ----------
    token_count : int
        Number of whitespace-separated tokens in the prompt.
    complexity : float
        Complexity score in [0, 1] from compute_prompt_features.

    Returns
    -------
    overhead_kwh : float
        Estimated extra energy in kWh.
    """
    # Base energy cost per token in kWh (tunable hyperparameter)
    base_per_token = 0.01  # e.g., 0.003 kWh per token

    # Complexity amplifies or shrinks overhead:
    # complexity 0.0 -> factor ~0.5, complexity 1.0 -> factor ~1.5
    complexity_factor = 1.0 + complexity

    overhead_kwh = token_count * base_per_token * complexity_factor
    return overhead_kwh

# --------------------------------------------------------
# 3. Load models
# --------------------------------------------------------

MODELS_DIR = Path("models")

try:
    lin_model, mlp_bundle = load_models(MODELS_DIR)
    mlp_scaler = mlp_bundle["scaler"]
    mlp_model = mlp_bundle["model"]
    models_loaded = True
except Exception as e:
    st.error(f" Could not load models from {MODELS_DIR}: {e}")
    models_loaded = False
    lin_model = None
    mlp_scaler = None
    mlp_model = None

# --------------------------------------------------------
# 4. Sidebar – model choice & info
# --------------------------------------------------------

st.sidebar.title("EcoFlux Settings")
st.sidebar.markdown(
    "Select which trained model EcoFlux should use for **base energy** predictions."
)

model_choice = st.sidebar.radio(
    "Choose prediction model:",
    ("Linear Regression (recommended)", "MLPRegressor (neural network)"),
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**About EcoFlux**  \n"
    "EcoFlux estimates the energy usage (kWh) of an ML training run based on:\n"
    "- model depth (layers)\n"
    "- training duration (hours)\n"
    "- compute intensity (GFLOPs/hour)\n"
    "- prompt length / complexity (prototype)\n\n"
    "It is an **educational tool**, not a precise carbon accounting system."
)

# --------------------------------------------------------
# 5. Main layout – title and description
# --------------------------------------------------------

st.title("🌿 EcoFlux: Machine Learning Energy Estimator")
st.caption("Making machine learning greener through transparency.")

st.markdown(
    """
EcoFlux helps you explore how **model design choices** and **prompt length**
affect estimated training energy consumption.  
Enter a prompt, adjust the sliders, and see both the original energy estimate
and a recommended lower-energy prompt.
"""
)

st.markdown("---")

# Two equal-width columns: left (inputs) and right (results)
col_left, col_right = st.columns([1, 1])

# --------------------------------------------------------
# 6. Left column – prompt + numeric inputs
# --------------------------------------------------------

with col_left:
    st.subheader("Prompt and Model Configuration")

    default_prompt = (
        "Role: You are a sustainability-focused ML assistant.\n"
        "Context: Explain the trade-offs between model size and energy use.\n"
        "Expectation: Give 3 concise bullet points suitable for students."
    )

    # Initialise once
    if "prompt_input" not in st.session_state:
        st.session_state.prompt_input = default_prompt

    # Callback used by the clear button
    def clear_prompt():
        st.session_state.prompt_input = ""

    # Text area is fully controlled by session_state
    st.text_area(
        "Enter your LLM prompt here:",
        key="prompt_input",
        height=200,
        help="This is the text the LLM would receive as input.",
    )

    # Clear / eraser button – uses callback
    st.button("Clear prompt", on_click=clear_prompt)

    # For later calculations, read from session_state
    prompt_text = st.session_state.prompt_input

    num_layers = st.slider(
        "Number of layers",
        min_value=2,
        max_value=24,
        value=8,
        step=1,
        help="Approximate depth of the model.",
    )
        
    training_hours = st.slider(
        "Training duration (hours)",
        min_value=0.5,
        max_value=24.0,
        value=6.0,
        step=0.5,
        help="How long the model is trained.",
    )

    flops_per_hour = st.slider(
        "Compute intensity (GFLOPs/hour)",
        min_value=10.0,
        max_value=300.0,
        value=120.0,
        step=5.0,
        help="Approximate computation per hour.",
    )

    st.markdown("")
    predict_button = st.button("Predict Energy & Recommend Prompt", type="primary")

# --------------------------------------------------------
# 7. Right column – predictions, recommendation & comparison
# --------------------------------------------------------

with col_right:
    st.subheader("Energy & Sustainability")

    if not models_loaded:
        st.warning("Models are not loaded. Please check the models directory.")
    elif not predict_button:
        st.info(
            "Enter a prompt, adjust the parameters on the left, "
            "and click **Predict Energy & Recommend Prompt**."
        )
    else:
        # -------- 7.1 Base energy from numeric model --------
        X_input = np.array([[num_layers, training_hours, flops_per_hour]])

        # ✅ LinearRegression: use raw (unscaled) features
        # ✅ MLPRegressor: use scaled features (mlp_scaler)
        if model_choice.startswith("Linear"):
            base_energy_raw = lin_model.predict(X_input)[0]
            model_used = "Linear Regression"
        else:
            X_scaled = mlp_scaler.transform(X_input)
            base_energy_raw = mlp_model.predict(X_scaled)[0]
            model_used = "MLPRegressor"

        # --- NEW: enforce non-negative (or a small physical lower-bound) ---
        MIN_ENERGY = 0.1  # or use the true min from your dataset
        base_energy = float(base_energy_raw)
        base_energy = max(MIN_ENERGY, base_energy)

        # Baseline reference configuration (e.g., 8 layers, 6h, 120 GFLOPs/h)
        baseline_X = np.array([[8, 6.0, 120.0]])

        # ✅ Again: no scaling for LinearRegression
        baseline_energy = lin_model.predict(baseline_X)[0]
        baseline_energy = max(0.0, float(baseline_energy))

        # -------- 7.2 Original prompt complexity & energy --------
        orig_tokens, orig_lines, orig_complexity = compute_prompt_features(prompt_text)

        # NEW: additive overhead in kWh based on tokens & complexity
        orig_overhead = prompt_overhead_kwh(orig_tokens, orig_complexity)
        orig_total_energy = base_energy + orig_overhead

        # -------- 7.3 Recommended simpler prompt --------
        improved_prompt = suggest_simpler_prompt(prompt_text)
        imp_tokens, imp_lines, imp_complexity = compute_prompt_features(improved_prompt)

        imp_overhead = prompt_overhead_kwh(imp_tokens, imp_complexity)
        imp_total_energy = base_energy + imp_overhead

        saving_kwh = orig_total_energy - imp_total_energy

        # -------- 7.4 Sustainability classification (original prompt) --------
        label, color, delta_vs_baseline = classify_sustainability(
            orig_total_energy, baseline_energy
        )

        # Main metric for original prompt
        st.metric(
            label="Original prompt energy estimate (kWh)",
            value=f"{orig_total_energy:.2f}",
            delta=f"{delta_vs_baseline:+.2f} kWh vs baseline",
        )

        st.markdown(
            f"**Sustainability rating:** "
            f"<span style='color:{color}; font-weight:bold;'>{label}</span>",
            unsafe_allow_html=True,
        )

        st.markdown(f"*Base model used for estimation: **{model_used}***")

 # -------- 7.4-bis Anomaly detection (IsolationForest) --------
        try:
            iso_model = load_iso_model(MODELS_DIR)

            status_text, status_color, iso_score = classify_anomaly(
                num_layers=num_layers,
                training_hours=training_hours,
                flops_per_hour=flops_per_hour,
                base_energy=base_energy,
                iso_model=iso_model,
                score_threshold_warn=-0.53,  # you can tune this
            )

            st.markdown(
                f"**Anomaly status (IsolationForest):** "
                f"<span style='color:{status_color}; font-weight:bold;'>{status_text}</span> "
                f"(score = {iso_score:.3f})",
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.info(f"Could not compute anomaly status: {e}")

        # -------- 7.5 Comparison table --------
        comparison_df = pd.DataFrame(
            {
                "Variant": ["Original prompt", "Recommended prompt"],
                "Tokens": [orig_tokens, imp_tokens],
                "Lines": [orig_lines, imp_lines],
                "Complexity score (0–1)": [
                    round(orig_complexity, 3),
                    round(imp_complexity, 3),
                ],
                "Prompt overhead (kWh)": [
                    round(orig_overhead, 3),
                    round(imp_overhead, 3),
                ],
                "Total energy (kWh)": [
                    round(orig_total_energy, 3),
                    round(imp_total_energy, 3),
                ],
            }
        )

        st.markdown("#### Original vs Recommended Prompt (Energy Comparison)")
        st.dataframe(comparison_df, use_container_width=True)

        if saving_kwh > 0:
            st.success(
                f"By using the recommended prompt, EcoFlux estimates a saving of "
                f"**{saving_kwh:.2f} kWh** for this configuration."
            )
        else:
            st.info(
                "The recommended prompt is already very concise — "
                "no additional energy savings are estimated."
            )

        # -------- 7.6 Show both prompts for inspection --------
        with st.expander("🔎 View prompts"):
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.markdown("**Original prompt**")
                st.code(prompt_text, language="markdown")
            with col_p2:
                st.markdown("**Recommended lower-energy prompt**")
                st.code(improved_prompt, language="markdown")

        st.markdown(
            """
            **Quick Tips:**

            - Shorter, clearer prompts usually reduce token count and energy overhead.  
            - Keeping role/context focused and avoiding repetition helps both clarity and sustainability.  
            - Model-side choices (layers, hours, FLOPs/hour) and prompt design *together*
            determine the final energy estimate.
            """
                    )
                        # -------- 7.6 Model visualisation + prompt-specific markers --------
        try:
            data_path = Path("data/energy_synthetic_structured.csv")
            df_energy = load_energy_data(data_path)

            # Common: features and actual target from synthetic dataset
            X_all = df_energy[["num_layers", "training_hours", "flops_per_hour"]]
            y_actual_all = df_energy["energy_kwh"].values

            # ----- LINEAR REGRESSION VIEW: 2D actual vs predicted -----
            if model_choice.startswith("Linear"):
                y_pred_all = lin_model.predict(X_all)

                # Create 2D figure/axis
                fig, ax = plt.subplots()

                # DATA POINTS — more visible (alpha=0.75)
                ax.scatter(
                    y_actual_all,
                    y_pred_all,
                    alpha=0.75,
                    color="#4C72B0",
                    label="Data points",
                )

                # BEST-FIT LINE (orange)
                m, b = np.polyfit(y_actual_all, y_pred_all, 1)
                x_line = np.linspace(y_actual_all.min(), y_actual_all.max(), 100)
                y_line = m * x_line + b
                ax.plot(
                    x_line,
                    y_line,
                    color="#F28E2B",
                    linewidth=2,
                    label="Linear regression best-fit line",
                )

                # Current configuration: treat base_energy as "actual-ish" x
                x_cfg = base_energy

                # ORIGINAL PROMPT (purple diamond)
                ax.scatter(
                    x_cfg,
                    orig_total_energy,
                    marker="D",
                    s=120,
                    color="#9467BD",
                    edgecolors="black",
                    linewidth=1,
                    label="Original prompt energy",
                    zorder=6,
                )

                # RECOMMENDED PROMPT (red square)
                ax.scatter(
                    x_cfg,
                    imp_total_energy,
                    marker="s",
                    s=120,
                    color="#D62728",
                    edgecolors="black",
                    linewidth=1,
                    label="Recommended prompt energy",
                    zorder=6,
                )

                # Make sure axes start at 0
                ax.set_xlim(left=0)
                ax.set_ylim(bottom=0)

                # ----- RECOMMENDED POINT DASHED GUIDES (TO AXES) -----
                ax.vlines(
                    x_cfg,
                    ymin=0,                 # from x-axis (0 on y)
                    ymax=imp_total_energy,
                    linestyles=":",
                    linewidth=1.2,
                    color="#6E6E6E",
                )
                ax.hlines(
                    imp_total_energy,
                    xmin=0,                 # from y-axis (0 on x)
                    xmax=x_cfg,
                    linestyles=":",
                    linewidth=1.2,
                    color="#6E6E6E",
                )

                ax.set_xlabel("Actual energy (kWh)")
                ax.set_ylabel("Predicted / prompt-adjusted energy (kWh)")
                ax.set_title("Linear Regression: Actual vs Predicted Energy")
                ax.legend(loc="upper left", frameon=True)

            # ----- MLP VIEW: 3D non-linear surface over (layers, FLOPs) -----
            else:
                # Grid ranges
                layers_vals = np.linspace(
                    df_energy["num_layers"].min(),
                    df_energy["num_layers"].max(),
                    20,
                )
                flops_vals = np.linspace(
                    df_energy["flops_per_hour"].min(),
                    df_energy["flops_per_hour"].max(),
                    20,
                )
                L_grid, F_grid = np.meshgrid(layers_vals, flops_vals)

                # Build grid for prediction
                X_grid = np.column_stack([
                    L_grid.ravel(),
                    np.full_like(L_grid.ravel(), training_hours),
                    F_grid.ravel(),
                ])
                X_grid_scaled = mlp_scaler.transform(X_grid)
                Z_pred = mlp_model.predict(X_grid_scaled).reshape(L_grid.shape)

                # Slightly smaller figure
                fig = plt.figure(figsize=(5, 4))
                ax = fig.add_subplot(111, projection="3d")

                # Pull the camera back to avoid cut-off
                ax.view_init(elev=25, azim=135)

                # -------------------------------------------------------
                # 3D SURFACE — transparent + lighter (MAIN FIX)
                # -------------------------------------------------------
                ax.plot_surface(
                    L_grid,
                    F_grid,
                    Z_pred,
                    cmap="cividis",      # cleaner & lighter colormap (optional)
                    alpha=0.45,          # MUCH lighter so markers remain visible
                    linewidth=0,
                    antialiased=True,
                )

                # Scatter real synthetic points
                ax.scatter(
                    df_energy["num_layers"],
                    df_energy["flops_per_hour"],
                    df_energy["energy_kwh"],
                    color="#4C72B0",
                    alpha=0.45,
                    s=15,
                    label="Observed energy (synthetic)",
                )

                # -------------------------------------------------------
                # Current config markers — easier to see
                # -------------------------------------------------------
                ax.scatter(
                    num_layers,
                    flops_per_hour,
                    orig_total_energy,
                    marker="D",
                    s=100,                 # larger for visibility
                    color="#9467BD",
                    edgecolors="black",
                    linewidth=1,
                    label="Original prompt energy",
                    zorder=10,
                )
                ax.scatter(
                    num_layers,
                    flops_per_hour,
                    imp_total_energy,
                    marker="s",
                    s=100,
                    color="#D62728",
                    edgecolors="black",
                    linewidth=1,
                    label="Recommended prompt energy",
                    zorder=10,
                )

                # Axis padding so nothing is cropped
                ax.set_xlim(
                    df_energy["num_layers"].min() - 1,
                    df_energy["num_layers"].max() + 1,
                )
                ax.set_ylim(
                    df_energy["flops_per_hour"].min() - 20,
                    df_energy["flops_per_hour"].max() + 20,
                )

                z_min = min(
                    df_energy["energy_kwh"].min(),
                    Z_pred.min(),
                    orig_total_energy,
                    imp_total_energy,
                )
                z_max = max(
                    df_energy["energy_kwh"].max(),
                    Z_pred.max(),
                    orig_total_energy,
                    imp_total_energy,
                )
                ax.set_zlim(z_min - 0.4, z_max + 0.6)

                # Labels & title
                ax.set_xlabel("Number of layers")
                ax.set_ylabel("Compute intensity\n(GFLOPs/h)", labelpad=8)
                ax.set_zlabel("Predicted / prompt-adjusted energy (kWh)")
                ax.set_title(
                    "MLPRegressor: Non-linear Energy Surface\n"
                    f"(training_hours fixed at {training_hours:.1f} h)"
                )
                ax.legend(loc="upper left")

            # Common Streamlit rendering
            st.markdown("#### Energy Curve and Prompt-Specific Estimates")
            fig.tight_layout(pad=0.5)
            st.pyplot(fig)

        except Exception as e:
            st.info(f"Could not draw energy curve from synthetic dataset: {e}")