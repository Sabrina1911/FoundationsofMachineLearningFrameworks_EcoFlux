import numpy as np
import pandas as pd
from pathlib import Path

import streamlit as st
import joblib
import matplotlib.pyplot as plt
import re

# --------------------------------------------------------
# EcoFlux: A Prompt-Aware Machine Learning Energy Estimator
# Streamlit GUI
#
# This version:
# - uses trained regression models (Linear / MLP)
# - adds prompt complexity at inference time (prototype)
# - recommends a shorter prompt
# - shows a 2D plot for Linear + a 3D surface for MLP
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
    lin_path = models_dir / "ecoflux_linear_regression.pkl"
    mlp_path = models_dir / "ecoflux_mlp_regressor.pkl"

    lin_model = joblib.load(lin_path)
    mlp_bundle = joblib.load(mlp_path)

    return lin_model, mlp_bundle


@st.cache_data
def load_energy_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    return pd.read_csv(csv_path)


def classify_sustainability(energy_kwh: float, baseline_kwh: float = 2.0):
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
    X_input_iso = np.array([[num_layers, training_hours, flops_per_hour, base_energy]])

    pred_label = iso_model.predict(X_input_iso)[0]      # 1 normal, -1 anomaly
    score = iso_model.score_samples(X_input_iso)[0]     # higher = more normal

    if pred_label == -1:
        status_text = "🔴 Anomalous configuration (out-of-distribution)"
        status_color = "red"
    else:
        if score < score_threshold_warn:
            status_text = "🟡 Unusual but not critical"
            status_color = "orange"
        else:
            status_text = "🟢 Normal configuration"
            status_color = "green"

    return status_text, status_color, score


def compute_prompt_features(prompt: str):
    word_tokens = prompt.split()
    word_token_count = len(word_tokens)

    newline_count = prompt.count("\n")
    token_count = word_token_count + newline_count

    lines = [ln for ln in prompt.splitlines() if ln.strip()]
    line_count = max(1, len(lines))

    # Optional: you computed this, but not using it now (kept for completeness)
    # avg_tokens_per_line = token_count / line_count

    length_component = min(1.0, token_count / 200.0)
    complexity_score = length_component

    return token_count, line_count, complexity_score


def suggest_simpler_prompt(prompt: str, max_tokens: int = 80):
    cleaned = " ".join(prompt.strip().split())
    if not cleaned:
        return cleaned

    tokens = cleaned.split()
    token_count = len(tokens)

    if token_count <= 10:
        return cleaned

    has_role = "Role:" in cleaned
    has_context = "Context:" in cleaned

    if has_role and has_context and token_count > max_tokens:
        try:
            role_start = cleaned.index("Role:")
            role_fragment = cleaned[role_start:]
            dot_idx = role_fragment.find(".")
            if dot_idx != -1:
                role_sentence = role_fragment[: dot_idx + 1]
            else:
                role_sentence = role_fragment
        except ValueError:
            role_sentence = "Role: You are a sustainability-focused AI assistant."

        simplified = (
            f"{role_sentence} "
            "Context: Summarise how large language models affect energy use, carbon footprint, "
            "and sustainability. "
            "Expectation: Explain trade-offs between model performance and compute cost, and "
            "mention renewable energy, efficient hardware, and responsible AI practices. "
            "Final task: End with 3–4 bullet-point takeaways linked to modern sustainability goals."
        )
        return " ".join(simplified.split())

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

    cleaned = " ".join(cleaned.split())
    tokens = cleaned.split()

    if len(tokens) <= max_tokens:
        return cleaned

    return " ".join(tokens[:max_tokens]) + " ..."


def prompt_overhead_kwh(token_count: int, complexity: float) -> float:
    base_per_token = 0.01
    complexity_factor = 1.0 + complexity
    return token_count * base_per_token * complexity_factor


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
    st.error(f"Could not load models from {MODELS_DIR}: {e}")
    models_loaded = False
    lin_model = None
    mlp_scaler = None
    mlp_model = None


# --------------------------------------------------------
# 4. Sidebar
# --------------------------------------------------------

st.sidebar.title("EcoFlux Settings")
st.sidebar.markdown("Select which trained model EcoFlux should use for **base energy** predictions.")

model_choice = st.sidebar.radio(
    "Choose prediction model:",
    ("Linear Regression (recommended)", "MLPRegressor (neural network)"),
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**About EcoFlux**  \n"
    "EcoFlux estimates energy usage (kWh) based on:\n"
    "- model depth (layers)\n"
    "- training duration (hours)\n"
    "- compute intensity (GFLOPs/hour)\n"
    "- prompt length / complexity (prototype)\n\n"
    "It is an **educational tool**, not a precise carbon accounting system."
)

# --------------------------------------------------------
# 5. Main layout
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
col_left, col_right = st.columns([1, 1])

# --------------------------------------------------------
# 6. Inputs (left)
# --------------------------------------------------------

with col_left:
    st.subheader("Prompt and Model Configuration")

    default_prompt = (
        "Role: You are a sustainability-focused ML assistant.\n"
        "Context: Explain the trade-offs between model size and energy use.\n"
        "Expectation: Give 3 concise bullet points suitable for students."
    )

    if "prompt_input" not in st.session_state:
        st.session_state.prompt_input = default_prompt

    def clear_prompt():
        st.session_state.prompt_input = ""

    st.text_area(
        "Enter your LLM prompt here:",
        key="prompt_input",
        height=200,
        help="This is the text the LLM would receive as input.",
    )
    st.button("Clear prompt", on_click=clear_prompt)

    prompt_text = st.session_state.prompt_input

    num_layers = st.slider("Number of layers", 2, 24, 8, 1)
    training_hours = st.slider("Training duration (hours)", 0.5, 24.0, 6.0, 0.5)
    flops_per_hour = st.slider("Compute intensity (GFLOPs/hour)", 10.0, 300.0, 120.0, 5.0)

    predict_button = st.button("Predict Energy & Recommend Prompt", type="primary")

# --------------------------------------------------------
# 7. Results (right)
# --------------------------------------------------------

with col_right:
    st.subheader("Energy & Sustainability")

    if not models_loaded:
        st.warning("Models are not loaded. Please check the models directory.")
    elif not predict_button:
        st.info("Enter inputs on the left and click **Predict Energy & Recommend Prompt**.")
    else:
        # 7.1 Base energy
        X_input = np.array([[num_layers, training_hours, flops_per_hour]])

        if model_choice.startswith("Linear"):
            base_energy_raw = lin_model.predict(X_input)[0]
            model_used = "Linear Regression"
        else:
            X_scaled = mlp_scaler.transform(X_input)
            base_energy_raw = mlp_model.predict(X_scaled)[0]
            model_used = "MLPRegressor"

        MIN_ENERGY = 0.1
        base_energy = max(MIN_ENERGY, float(base_energy_raw))

        baseline_X = np.array([[8, 6.0, 120.0]])
        baseline_energy = max(0.0, float(lin_model.predict(baseline_X)[0]))

        # 7.2 Prompt overheads
        orig_tokens, orig_lines, orig_complexity = compute_prompt_features(prompt_text)
        orig_overhead = prompt_overhead_kwh(orig_tokens, orig_complexity)
        orig_total_energy = base_energy + orig_overhead

        improved_prompt = suggest_simpler_prompt(prompt_text)
        imp_tokens, imp_lines, imp_complexity = compute_prompt_features(improved_prompt)
        imp_overhead = prompt_overhead_kwh(imp_tokens, imp_complexity)
        imp_total_energy = base_energy + imp_overhead

        saving_kwh = orig_total_energy - imp_total_energy

        # 7.3 Sustainability label
        label, color, delta_vs_baseline = classify_sustainability(orig_total_energy, baseline_energy)

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

        # 7.4 IsolationForest anomaly
        try:
            iso_model = load_iso_model(MODELS_DIR)
            status_text, status_color, iso_score = classify_anomaly(
                num_layers=num_layers,
                training_hours=training_hours,
                flops_per_hour=flops_per_hour,
                base_energy=base_energy,
                iso_model=iso_model,
                score_threshold_warn=-0.53,
            )

            st.markdown(
                f"**Anomaly status (IsolationForest):** "
                f"<span style='color:{status_color}; font-weight:bold;'>{status_text}</span> "
                f"(score = {iso_score:.3f})",
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.info(f"Could not compute anomaly status: {e}")

        # 7.5 Comparison table
        comparison_df = pd.DataFrame(
            {
                "Variant": ["Original prompt", "Recommended prompt"],
                "Tokens": [orig_tokens, imp_tokens],
                "Lines": [orig_lines, imp_lines],
                "Complexity score (0–1)": [round(orig_complexity, 3), round(imp_complexity, 3)],
                "Prompt overhead (kWh)": [round(orig_overhead, 3), round(imp_overhead, 3)],
                "Total energy (kWh)": [round(orig_total_energy, 3), round(imp_total_energy, 3)],
            }
        )

        st.markdown("#### Original vs Recommended Prompt (Energy Comparison)")
        st.dataframe(comparison_df, use_container_width=True)

        if saving_kwh > 0:
            st.success(f"Estimated saving using the recommended prompt: **{saving_kwh:.2f} kWh**")
        else:
            st.info("Recommended prompt is already very concise — no extra savings estimated.")

        with st.expander("🔎 View prompts"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Original prompt**")
                st.code(prompt_text, language="markdown")
            with c2:
                st.markdown("**Recommended lower-energy prompt**")
                st.code(improved_prompt, language="markdown")

        st.markdown(
            """
**Quick Tips:**
- Shorter, clearer prompts reduce token count and energy overhead.
- Avoid repetition; keep role/context focused.
- Model-side choices and prompt design *together* determine energy.
"""
        )

        # --------------------------------------------------------
        # 7.6 Model visualisation + prompt markers
        # --------------------------------------------------------
        try:
            data_path = Path("data/energy_synthetic_structured.csv")
            df_energy = load_energy_data(data_path)

            X_all = df_energy[["num_layers", "training_hours", "flops_per_hour"]]
            y_actual_all = df_energy["energy_kwh"].values

            # ========== LINEAR: 2D Actual vs Predicted ==========
            if model_choice.startswith("Linear"):
                y_pred_all = lin_model.predict(X_all)

                fig, ax = plt.subplots()

                ax.scatter(
                    y_actual_all,
                    y_pred_all,
                    alpha=0.75,
                    label="Data points",
                )

                m, b = np.polyfit(y_actual_all, y_pred_all, 1)
                x_line = np.linspace(y_actual_all.min(), y_actual_all.max(), 100)
                y_line = m * x_line + b
                ax.plot(x_line, y_line, linewidth=2, label="Linear regression best-fit line")

                # Use the clamped base_energy (always visible/meaningful)
                x_cfg = float(base_energy)
                y_cfg = x_cfg

                # Tiny VISUAL offsets so markers don't sit under the densest cloud
                x_offset = 0.6
                y_offset = 0.8

                # Marker positions (visual only)
                x_base = x_cfg + x_offset
                y_base = y_cfg + y_offset

                x_org = x_cfg + x_offset
                y_org = orig_total_energy + y_offset

                x_rec = x_cfg + x_offset
                y_rec = imp_total_energy + y_offset

                # Markers (high zorder)
                ax.scatter(x_org, y_org, marker="D", s=190, color="#9467BD",
                           edgecolors="black", linewidth=1.2,
                           label="Original prompt energy", zorder=21)

                ax.scatter(x_rec, y_rec, marker="s", s=190, color="#D62728",
                           edgecolors="black", linewidth=1.2,
                           label="Recommended prompt energy", zorder=22)

                # Dotted guides from recommended marker to axes-floor (works with negatives)
                # Compute limits first, then guides use those minima
                xmax = max(float(y_actual_all.max()), x_rec) * 1.05
                ymax = max(float(y_pred_all.max()), y_org, y_rec) * 1.05
                xmin = min(float(y_actual_all.min()), x_rec) * 1.05
                ymin = min(float(y_pred_all.min()), y_org, y_rec) * 1.05

                # Zoom / frame so lower-left is readable
                ax.set_xlim(min(-2, xmin), max(25, xmax))
                ax.set_ylim(min(-15, ymin), max(25, ymax))

                x_min = ax.get_xlim()[0]
                y_min = ax.get_ylim()[0]

                ax.vlines(x_rec, ymin=y_min, ymax=y_rec, linestyles=":", linewidth=1.6, color="#6E6E6E", zorder=5)
                ax.hlines(y_rec, xmin=x_min, xmax=x_rec, linestyles=":", linewidth=1.6, color="#6E6E6E", zorder=5)

                # Annotations (guarantees visibility)
                ax.annotate("Original", (x_org, y_org), xytext=(12, 14), textcoords="offset points",
                            arrowprops=dict(arrowstyle="->", color="#9467BD"), fontsize=9)
                ax.annotate("Recommended", (x_rec, y_rec), xytext=(12, -18), textcoords="offset points",
                            arrowprops=dict(arrowstyle="->", color="#D62728"), fontsize=9)

                # Reference axes at 0
                ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
                ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6)

                ax.set_xlabel("Actual energy (kWh)")
                ax.set_ylabel("Predicted / prompt-adjusted energy (kWh)")
                ax.set_title("Linear Regression: Actual vs Predicted Energy")

                # Legend away from the bottom-left cluster
                ax.legend(loc="upper right", frameon=True)

            # ========== MLP: 3D surface ==========
            else:
                layers_vals = np.linspace(df_energy["num_layers"].min(), df_energy["num_layers"].max(), 20)
                flops_vals = np.linspace(df_energy["flops_per_hour"].min(), df_energy["flops_per_hour"].max(), 20)

                L_grid, F_grid = np.meshgrid(layers_vals, flops_vals)

                X_grid = np.column_stack([
                    L_grid.ravel(),
                    np.full_like(L_grid.ravel(), training_hours),
                    F_grid.ravel(),
                ])
                X_grid_scaled = mlp_scaler.transform(X_grid)
                Z_pred = mlp_model.predict(X_grid_scaled).reshape(L_grid.shape)

                fig = plt.figure(figsize=(5.2, 4.2))
                ax = fig.add_subplot(111, projection="3d")
                ax.view_init(elev=25, azim=135)

                ax.plot_surface(
                    L_grid, F_grid, Z_pred,
                    cmap="cividis",
                    alpha=0.30,
                    linewidth=0,
                    antialiased=True,
                )

                ax.scatter(
                    df_energy["num_layers"],
                    df_energy["flops_per_hour"],
                    df_energy["energy_kwh"],
                    alpha=0.20,
                    s=12,
                    label="Observed energy (synthetic)",
                    depthshade=False,
                )

                # lift markers slightly above surface
                z_pad = 0.02 * (Z_pred.max() - Z_pred.min())
                z_orig = orig_total_energy + z_pad
                z_imp = imp_total_energy + z_pad

                ax.scatter(
                    num_layers, flops_per_hour, z_orig,
                    marker="D", s=130, color="#9467BD",
                    edgecolors="black", linewidth=1.2,
                    label="Original prompt energy",
                    depthshade=False,
                )

                ax.scatter(
                    num_layers, flops_per_hour, z_imp,
                    marker="s", s=130, color="#D62728",
                    edgecolors="black", linewidth=1.2,
                    label="Recommended prompt energy",
                    depthshade=False,
                )

                # Set limits first, then build 3D guide lines using x0,y0,z0 anchors
                ax.set_xlim(df_energy["num_layers"].min() - 1, df_energy["num_layers"].max() + 1)
                ax.set_ylim(df_energy["flops_per_hour"].min() - 20, df_energy["flops_per_hour"].max() + 20)

                z_min = min(df_energy["energy_kwh"].min(), Z_pred.min(), z_orig, z_imp)
                z_max = max(df_energy["energy_kwh"].max(), Z_pred.max(), z_orig, z_imp)
                ax.set_zlim(z_min - 0.4, z_max + 0.6)

                z0 = ax.get_zlim()[0]
                x0 = ax.get_xlim()[0]
                y0 = ax.get_ylim()[0]

                # 3D guide lines for recommended point
                ax.plot([num_layers, num_layers], [flops_per_hour, flops_per_hour], [z0, z_imp],
                        linestyle=":", linewidth=1.4, color="#6E6E6E")
                ax.plot([x0, num_layers], [flops_per_hour, flops_per_hour], [z0, z0],
                        linestyle=":", linewidth=1.4, color="#6E6E6E")
                ax.plot([num_layers, num_layers], [y0, flops_per_hour], [z0, z0],
                        linestyle=":", linewidth=1.4, color="#6E6E6E")

                ax.set_xlabel("Number of layers")
                ax.set_ylabel("Compute intensity (GFLOPs/h)", labelpad=8)
                ax.set_zlabel("Predicted / prompt-adjusted energy (kWh)")
                ax.set_title(
                    "MLPRegressor: Non-linear Energy Surface\n"
                    f"(training_hours fixed at {training_hours:.1f} h)"
                )

                ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98))

            st.markdown("#### Energy Curve and Prompt-Specific Estimates")
            fig.tight_layout(pad=0.5)
            st.pyplot(fig)

        except Exception as e:
            st.info(f"Could not draw energy curve from synthetic dataset: {e}")
