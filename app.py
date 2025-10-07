import sys
import types
sys.modules["pyaudioop"] = types.ModuleType("pyaudioop")

import gradio as gr
import pandas as pd
import numpy as np
import pickle
import os

def load_model():
    try:
        with open("catboost_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        raise Exception("Model file 'catboost_model.pkl' not found")
    except Exception as e:
        raise Exception(f"Failed to load model: {e}")

# Load model
model = load_model()

def predict_risk(Compressiontime, IntraopNTG, PreRaddiam, SRratio,
                 Heparincategory, Punctureattempts, Priorradpunctures):

    try:
        # Input validation
        if any(v is None for v in [Compressiontime, IntraopNTG, PreRaddiam, SRratio]):
            return "❌ Please fill in all required numerical parameters"
        
        # Create DataFrame
        df = pd.DataFrame([{
            "Compressiontime": float(Compressiontime),
            "Intraoperativenitroglycerindose": float(IntraopNTG),
            "PreRaddiam": float(PreRaddiam) / 10,  # Convert mm to cm
            "SRratio": float(SRratio),
            "Heparincategory": int(Heparincategory),
            "Punctureattempts": int(Punctureattempts),
            "History of prior radial artery catheterization": int(Priorradpunctures)
        }])

        # Check if standardization files exist
        if not (os.path.exists("feature_means.csv") and os.path.exists("feature_stds.csv")):
            return "❌ Standardization parameter files not found"
            
        # Load standardization parameters
        means = pd.read_csv("feature_means.csv", index_col=0).squeeze()
        stds = pd.read_csv("feature_stds.csv", index_col=0).squeeze()

        # Standardize only numerical features
        num_features = ['Compressiontime', 'Intraoperativenitroglycerindose', 'PreRaddiam', 'SRratio']
        df[num_features] = (df[num_features] - means[num_features]) / stds[num_features]

        # Model prediction
        prob = model.predict_proba(df)[0][1]
        
        # Risk stratification and recommendations
        if prob < 0.10:
            risk_level = "Low Risk"
            color = "🟢"
            suggestion = "Routine care recommended"
        elif prob < 0.30:
            risk_level = "Medium Risk"
            color = "🟡"
            suggestion = "Enhanced post-operative monitoring advised"
        else:
            risk_level = "High Risk"
            color = "🔴"
            suggestion = "Preventive measures and close monitoring required"
            
        result = f"""
{color} **Prediction Result: {risk_level}**

**Radial Artery Occlusion Probability:** {prob * 100:.2f}%  

**Clinical Recommendation:** {suggestion}

---
*Prediction based on CatBoost machine learning model - For reference only*
"""
        return result

    except Exception as e:
        return f"❌ Prediction failed: {str(e)}"

def reset_inputs():
    return [120, 200, 2.5, 0.6, "1", "1", "0"]

# === Gradio Interface ===
with gr.Blocks(theme=gr.themes.Soft(), title="RAO Risk Calculator") as demo:
    gr.Markdown("""
    # 🌡 Radial Artery Occlusion (RAO) Risk Calculator
    *Machine learning-based prediction of radial artery occlusion risk following transradial procedures*
    """)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Clinical Parameters")
            compression_time = gr.Number(label="Compression Time (minutes)", value=120, minimum=30, maximum=400, step=5)
            intraop_ntg = gr.Number(label="Intraoperative Nitroglycerin Dose (μg)", value=200, minimum=0, maximum=900, step=50)
            pre_rad_diam = gr.Number(label="Pre-procedural Radial Artery Diameter (mm)", value=2.5, minimum=0.5, maximum=3.8, step=0.1)
            sr_ratio = gr.Number(label="Sheath-to-Artery Ratio", value=0.6, minimum=0.1, maximum=1.5, step=0.05)
            
        with gr.Column():
            gr.Markdown("### Categorical Variables")
            heparin_category = gr.Radio(choices=[("≤5000 IU", "1"), ("≥5000 IU", "2")], label="Heparin Category", value="1")
            puncture_attempts = gr.Radio(choices=[("Single Puncture", "1"), ("Multiple Punctures", "2")], label="Puncture Attempts", value="1")
            prior_rad_punctures = gr.Radio(choices=[("No", "0"), ("Yes", "1")], label="Prior Radial Catheterization", value="0")

            with gr.Row():
                predict_btn = gr.Button("🚀 Calculate RAO Risk", variant="primary")
                reset_btn = gr.Button("🔄 Reset", variant="secondary")
    
    output_text = gr.Markdown(label="Prediction Result", value="Enter parameters and click calculate...")

    predict_btn.click(fn=predict_risk, 
                      inputs=[compression_time, intraop_ntg, pre_rad_diam, sr_ratio,
                              heparin_category, puncture_attempts, prior_rad_punctures],
                      outputs=output_text)

    reset_btn.click(fn=reset_inputs,
                    outputs=[compression_time, intraop_ntg, pre_rad_diam, sr_ratio,
                             heparin_category, puncture_attempts, prior_rad_punctures])

    gr.Markdown("""
    ---
    *This tool uses machine learning for prediction. Results are for reference only and should not replace clinical judgment.*
    """)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # ✅ Render uses PORT environment variable
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)












