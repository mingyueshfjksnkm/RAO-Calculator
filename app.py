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
            "PreRaddiam": float(PreRaddiam)/10,  # Convert to cm
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

# Create interface
with gr.Blocks(theme=gr.themes.Soft(), title="RAO Risk Calculator") as demo:
    gr.Markdown("""
    # 🌡 Radial Artery Occlusion (RAO) Risk Calculator
    
    *Machine learning-based prediction of radial artery occlusion risk following transradial procedures*
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Clinical Parameters")
            compression_time = gr.Number(
                label="Compression Time (minutes)",
                value=120,
                minimum=30,
                maximum=400,
                step=5,
                info="Typically 30-400 minutes"
            )
            intraop_ntg = gr.Number(
                label="Intraoperative Nitroglycerin Dose (μg)",
                value=200,
                minimum=0,
                maximum=900,
                step=50,
                info="Common dose: 0-900μg"
            )
            pre_rad_diam = gr.Number(
                label="Pre-procedural Radial Artery Diameter (mm)",
                value=2.5,
                minimum=0.5,
                maximum=3.8,
                step=0.1,
                info="Ultrasound measurement"
            )
            sr_ratio = gr.Number(
                label="Sheath-to-Artery Ratio",
                value=0.6,
                minimum=0.1,
                maximum=1.5,
                step=0.05,
                info="Sheath outer diameter / artery diameter"
            )
            
        with gr.Column():
            gr.Markdown("### Categorical Variables")
            heparin_category = gr.Radio(
                choices=[("≤5000 IU", "1"), ("≥5000 IU", "2")],
                label="Heparin Category",
                value="1",
                info="Anticoagulation dosage"
            )
            puncture_attempts = gr.Radio(
                choices=[("Single Puncture", "1"), ("Multiple Punctures", "2")],
                label="Puncture Attempts",
                value="1",
                info="Number of attempts before success"
            )
            prior_rad_punctures = gr.Radio(
                choices=[("No", "0"), ("Yes", "1")],
                label="History of Prior Radial Artery Catheterization",
                value="0",
                info="Previous ipsilateral radial procedures"
            )
            
            with gr.Row():
                predict_btn = gr.Button("🚀 Calculate RAO Risk", variant="primary", size="lg")
                reset_btn = gr.Button("🔄 Reset", variant="secondary")
    
    with gr.Row():
        output_text = gr.Markdown(
            label="Prediction Result",
            value="Enter parameters and click calculate..."
        )
    
    # Examples section
    with gr.Accordion("📋 Example Inputs", open=False):
        gr.Markdown("""
        **Example 1 - Low Risk Case:**
        - Compression Time: 120 minutes
        - Nitroglycerin: 200μg  
        - Radial Diameter: 2.5mm
        - Sheath-to-Artery Ratio: 0.6
        - Heparin: ≤5000 IU
        - Puncture: Single
        - No Prior History
        
        **Example 2 - High Risk Case:**
        - Compression Time: 180 minutes
        - Nitroglycerin: 100μg
        - Radial Diameter: 2.0mm  
        - Sheath-to-Artery Ratio: 0.8
        - Heparin: ≤5000 IU
        - Puncture: Multiple
        - Prior History Present
        """)
        
        examples = gr.Examples(
            examples=[
                [120, 200, 2.5, 0.6, "1", "1", "0"],
                [180, 100, 2.0, 0.8, "1", "2", "1"],
                [90, 300, 3.0, 0.5, "2", "1", "0"]
            ],
            inputs=[
                compression_time, intraop_ntg, pre_rad_diam, sr_ratio,
                heparin_category, puncture_attempts, prior_rad_punctures
            ],
            label="Quick Fill Examples"
        )
    
    # Event handlers
    predict_btn.click(
        fn=predict_risk,
        inputs=[
            compression_time, intraop_ntg, pre_rad_diam, sr_ratio,
            heparin_category, puncture_attempts, prior_rad_punctures
        ],
        outputs=output_text
    )
    
    reset_btn.click(
        fn=reset_inputs,
        outputs=[
            compression_time, intraop_ntg, pre_rad_diam, sr_ratio,
            heparin_category, puncture_attempts, prior_rad_punctures
        ]
    )

    # Footer
    gr.Markdown("""
    ---
    *This tool uses machine learning for prediction. Results are for reference only and should not replace clinical judgment.*
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0" if os.getenv('SPACE_ID') else "127.0.0.1",
        share=False
    )










