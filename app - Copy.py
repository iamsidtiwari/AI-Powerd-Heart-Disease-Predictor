import gradio as gr
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------
# 1. LOAD DATASET AND TRAIN MODEL
# -------------------------------------------------------

print("Loading dataset and training model...")

df = pd.read_csv('heart.csv')

# Separate input features and target variable
X = df.drop('target', axis=1)
y = df['target']

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save trained model
joblib.dump(model, 'heart_model.pkl')

print("Model trained and saved successfully.")


# -------------------------------------------------------
# 2. HEART DISEASE PREDICTION FUNCTION
# -------------------------------------------------------

def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal):
    """
    Generates a detailed heart disease risk prediction report
    based on patient vitals and clinical parameters.
    """

    features = np.array([[
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal
    ]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    risk_score = probability[1] * 100

    # Determine risk category
    if risk_score >= 70:
        result = "HIGH RISK - HEART DISEASE DETECTED"
        color = "#ff4444"
        advice = "Immediate consultation with a cardiologist is recommended."
        severity = "Critical"
    elif risk_score >= 40:
        result = "MODERATE RISK - BORDERLINE"
        color = "#ffaa00"
        advice = "Regular health checkups and monitoring are advised."
        severity = "Moderate"
    else:
        result = "LOW RISK - NO HEART DISEASE LIKELY"
        color = "#44aa44"
        advice = "Maintain a healthy lifestyle and routine checkups."
        severity = "Low"

    # Professional Report (HTML)
    report = f"""
<div style='border: 2px solid {color}; padding: 20px; border-radius: 10px; background: #f8f9fa;'>
    <h2 style='color: {color}; text-align: center;'>{result}</h2>
    
    <div style='text-align: center; margin: 20px 0;'>
        <div style='font-size: 24px; font-weight: bold; color: {color};'>
            Risk Score: {risk_score:.1f}%
        </div>
        <div style='color: #555;'>Severity Level: {severity}</div>
    </div>

    <div style='background: white; padding: 15px; border-radius: 8px; margin: 15px 0;'>
        <h3 style='color: #333;'>Medical Recommendation</h3>
        <p style='font-size: 16px;'>{advice}</p>
    </div>
    
    <div style='background: white; padding: 15px; border-radius: 8px;'>
        <h3 style='color: #333;'>Patient Vitals Overview</h3>
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px;'>
            <div><strong>Age:</strong> {age} years</div>
            <div><strong>Gender:</strong> {'Male' if sex == 1 else 'Female'}</div>
            <div><strong>Resting Blood Pressure:</strong> {trestbps} mmHg</div>
            <div><strong>Cholesterol:</strong> {chol} mg/dl</div>
            <div><strong>Maximum Heart Rate:</strong> {thalach} bpm</div>
            <div><strong>ST Depression:</strong> {oldpeak}</div>
        </div>
    </div>
</div>
    """

    return report


# -------------------------------------------------------
# 3. BUILD PROFESSIONAL WEB INTERFACE USING GRADIO
# -------------------------------------------------------

print("Building professional web interface...")

custom_css = """
.gradio-container {
    font-family: 'Arial', sans-serif;
}
.header {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #4a69bd 0%, #6a89cc 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 20px;
}
.section {
    background: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    margin: 10px 0;
}
"""

with gr.Blocks(title="Heart Disease Prediction System", css=custom_css) as demo:

    # Header
    with gr.Column(elem_classes=["header"]):
        gr.Markdown("# Heart Disease Prediction System")
        gr.Markdown("### AI-Based Clinical Risk Assessment Tool")
        gr.Markdown("Enter patient clinical parameters for analysis.")

    # Patient Information Section
    with gr.Column(elem_classes=["section"]):
        gr.Markdown("## Patient Information")

        with gr.Row():
            age = gr.Slider(1, 100, value=50, label="Age")
            sex = gr.Radio(["Female", "Male"], label="Gender", value="Female")
            cp = gr.Dropdown([0, 1, 2, 3], label="Chest Pain Type", value=0)

        with gr.Row():
            trestbps = gr.Slider(50, 200, value=120, label="Resting Blood Pressure (mmHg)")
            chol = gr.Slider(100, 600, value=200, label="Cholesterol (mg/dl)")
            thalach = gr.Slider(60, 220, value=150, label="Maximum Heart Rate Achieved")

        with gr.Row():
            fbs = gr.Radio(["No", "Yes"], label="Fasting Blood Sugar > 120 mg/dl?", value="No")
            restecg = gr.Dropdown([0, 1, 2], label="Resting ECG Result", value=0)
            exang = gr.Radio(["No", "Yes"], label="Exercise-Induced Angina?", value="No")

        with gr.Row():
            oldpeak = gr.Slider(0.0, 6.0, value=1.0, step=0.1, label="ST Depression")
            slope = gr.Dropdown([0, 1, 2], label="ST Slope", value=1)
            ca = gr.Slider(0, 4, value=0, label="Major Vessels (0-4)")
            thal = gr.Dropdown([1, 2, 3], label="Thalassemia Type", value=2)

    # Buttons
    with gr.Column(elem_classes=["section"]):
        with gr.Row():
            predict_btn = gr.Button("Analyze Risk", variant="primary")
            clear_btn = gr.Button("Clear All", variant="secondary")

    # Output Section
    with gr.Column(elem_classes=["section"]):
        gr.Markdown("## Prediction Results")
        output = gr.HTML(label="Risk Assessment Report")

    # Example Profiles
    with gr.Column(elem_classes=["section"]):
        gr.Markdown("## Example Profiles")
        gr.Markdown("""
**High Risk Profile:**  
Age: 63, Male, BP: 145, Cholesterol: 233, Chest Pain: 3, Max HR: 150

**Low Risk Profile:**  
Age: 45, Female, BP: 110, Cholesterol: 180, Chest Pain: 0, Max HR: 175
""")

    # Predict Button Function
    def on_predict_click(age, sex, cp, trestbps, chol, thalach, fbs,
                         restecg, exang, oldpeak, slope, ca, thal):

        sex_num = 1 if sex == "Male" else 0
        fbs_num = 1 if fbs == "Yes" else 0
        exang_num = 1 if exang == "Yes" else 0

        return predict_heart_disease(
            age, sex_num, cp, trestbps, chol, fbs_num,
            restecg, thalach, exang_num, oldpeak, slope, ca, thal
        )

    # Clear Button Function
    def clear_all():
        return [
            50, "Female", 0, 120, 200, 150, "No",
            0, "No", 1.0, 1, 0, 2, ""
        ]

    # Button Connections
    predict_btn.click(
        fn=on_predict_click,
        inputs=[age, sex, cp, trestbps, chol, thalach,
                fbs, restecg, exang, oldpeak, slope, ca, thal],
        outputs=output
    )

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[age, sex, cp, trestbps, chol, thalach,
                 fbs, restecg, exang, oldpeak, slope, ca, thal, output]
    )

print("Launching web application...")
demo.launch(share=True, debug=True)
