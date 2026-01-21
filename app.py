
import gradio as gr
import pandas as pd
import pickle
import numpy as np

MODEL_PATH = "diabetes_rf_pipeline.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

DEFAULT_COLS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
EXPECTED_COLS = list(getattr(model, "feature_names_in_", DEFAULT_COLS))

ZERO_AS_MISSING_COLS = {"Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"}


def _to_float(x, default=0.0):
    if x is None:
        return float(default)
    try:
        return float(x)
    except Exception:
        return float(default)


def _build_input_df(pregnancies, glucose, bp, skin, insulin, bmi, dpf, age, treat_zero_as_missing):
    pregnancies = int(round(_to_float(pregnancies, 0)))
    age = int(round(_to_float(age, 30)))

    glucose = _to_float(glucose, 120)
    bp = _to_float(bp, 70)
    skin = _to_float(skin, 20)
    insulin = _to_float(insulin, 80)
    bmi = _to_float(bmi, 28.0)
    dpf = _to_float(dpf, 0.35)

    row = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": bp,
        "SkinThickness": skin,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }

    input_df = pd.DataFrame([[row.get(c, np.nan) for c in EXPECTED_COLS]], columns=EXPECTED_COLS)

    if treat_zero_as_missing:
        for c in input_df.columns:
            if c in ZERO_AS_MISSING_COLS:
                input_df[c] = input_df[c].replace(0, np.nan)

    return input_df


def predict_diabetes(pregnancies, glucose, bp, skin, insulin, bmi, dpf, age, treat_zero_as_missing):
    input_df = _build_input_df(pregnancies, glucose, bp, skin, insulin, bmi, dpf, age, treat_zero_as_missing)

    pred_class = int(model.predict(input_df)[0])

    prob_text = ""
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(input_df)[0][1])
        prob_text = f" | Probability(Diabetes): {prob*100:.2f}%"

    label = "❌Diabetes (Outcome=1)\n" if pred_class == 1 else "✅Not Diabetes (Outcome=0)\n"
    return input_df, f"Prediction: {label}{prob_text}"


with gr.Blocks(
    css="""
    /* Make sliders/labels more compact */
    .gradio-container {max-width: 980px !important;}
    .block {padding: 10px !important;}
    label {margin-bottom: 4px !important;}
    """
) as demo:
    gr.Markdown("## Diabetes Prediction Demo (Random Forest PKL Pipeline)")

    gr.Markdown(
        "**Ranges (min–max):** Pregnancies 0–17 | Glucose 0–200 | BloodPressure 0–180 | "
        "SkinThickness 0–100 | Insulin 0–900 | BMI 0–70 | DPF 0–3 | Age 20–90 | Outcome 0/1"
    )

    treat_zero_as_missing = gr.Checkbox(
        value=True,
        label="Treat 0 as missing (turn OFF if training did NOT replace 0 with NaN)"
    )

    # Two-column layout:
    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=320):
            pregnancies = gr.Slider(0, 17, step=1, label="Pregnancies", value=1)
            glucose = gr.Slider(0, 200, step=1, label="Glucose", value=120)
            bp = gr.Slider(0, 180, step=1, label="BloodPressure", value=70)
            skin = gr.Slider(0, 100, step=1, label="SkinThickness", value=20)

        with gr.Column(scale=1, min_width=320):
            insulin = gr.Slider(0, 900, step=1, label="Insulin", value=80)
            bmi = gr.Slider(0, 70, step=0.1, label="BMI", value=28.0)
            dpf = gr.Slider(0.0, 3.0, step=0.01, label="DiabetesPedigreeFunction", value=0.35)
            age = gr.Slider(20, 90, step=1, label="Age", value=30)

    preview_df = gr.Dataframe(label="Input Row Sent to Model", interactive=False)

    def update_preview(p, g, b, s, i, bm, dp, a, tz):
        return _build_input_df(p, g, b, s, i, bm, dp, a, tz)

    for comp in [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age, treat_zero_as_missing]:
        comp.change(
            fn=update_preview,
            inputs=[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age, treat_zero_as_missing],
            outputs=preview_df,
            queue=False
        )

    with gr.Row():
        btn = gr.Button("Predict", scale=1)
        out_text = gr.Textbox(label="Result", lines=2, scale=2)

    btn.click(
        fn=predict_diabetes,
        inputs=[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age, treat_zero_as_missing],
        outputs=[preview_df, out_text],
    )

demo.launch(share=True)

#AjShihab

