# app.py — Streamlit 版 RAO 风险预测计算器
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ---------------------------
# 模型加载
# ---------------------------
@st.cache_resource
def load_model():
    try:
        with open("catboost_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'catboost_model.pkl' not found.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

model = load_model()

# ---------------------------
# 标准化参数加载
# ---------------------------
def load_scaler_params():
    if not (os.path.exists("feature_means.csv") and os.path.exists("feature_stds.csv")):
        st.error("❌ Standardization parameter files not found.")
        st.stop()
    means = pd.read_csv("feature_means.csv", index_col=0).squeeze()
    stds = pd.read_csv("feature_stds.csv", index_col=0).squeeze()
    return means, stds

means, stds = load_scaler_params()

# ---------------------------
# 页面设置
# ---------------------------
st.set_page_config(
    page_title="RAO Risk Calculator",
    layout="wide",
    page_icon="🌡"
)

st.title("🌡 Radial Artery Occlusion (RAO) Risk Calculator")
st.markdown("""
*Machine learning-based prediction of radial artery occlusion risk following transradial procedures.*
---
""")

# ---------------------------
# 输入部分
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Clinical Parameters")
    Compressiontime = st.number_input("Compression Time (minutes)", 30.0, 1200.0, 120.0, step=5.0)
    IntraopNTG = st.number_input("Intraoperative Nitroglycerin Dose (μg)", 0.0, 2500.0, 200.0, step=50.0)
    PreRaddiam = st.number_input("Pre-procedural Radial Artery Diameter (mm)", 0.5, 7.0, 2.5, step=0.1)
    SRratio = st.number_input("Sheath-to-Artery Ratio", 0.1, 2.0, 0.6, step=0.05)

with col2:
    st.subheader("Categorical Variables")
    Heparincategory = st.radio("Heparin Category", options=["1", "2"], format_func=lambda x: "≤5000 IU" if x=="1" else "≥5000 IU")
    Punctureattempts = st.radio("Puncture Attempts", options=["1", "2"], format_func=lambda x: "Single" if x=="1" else "Multiple")
    Priorradpunctures = st.radio("History of Prior Radial Artery Catheterization", options=["0", "1"], format_func=lambda x: "No" if x=="0" else "Yes")

# ---------------------------
# 预测逻辑
# ---------------------------
def predict_risk(Compressiontime, IntraopNTG, PreRaddiam, SRratio,
                 Heparincategory, Punctureattempts, Priorradpunctures):
    try:
        df = pd.DataFrame([{
            "Compressiontime": Compressiontime,
            "Intraoperativenitroglycerindose": IntraopNTG,
            "PreRaddiam": PreRaddiam / 10,  # 转换成 cm
            "SRratio": SRratio,
            "Heparincategory": int(Heparincategory),
            "Punctureattempts": int(Punctureattempts),
            "History of prior radial artery catheterization": int(Priorradpunctures)
        }])
        num_features = ['Compressiontime', 'Intraoperativenitroglycerindose', 'PreRaddiam', 'SRratio']
        df[num_features] = (df[num_features] - means[num_features]) / stds[num_features]

        prob = model.predict_proba(df)[0][1]

        if prob < 0.05:
            risk_level, color, suggestion = "Low Risk", "🟢", "Routine care recommended"
        elif prob < 0.15:
            risk_level, color, suggestion = "Medium Risk", "🟡", "Enhanced post-operative monitoring advised"
        else:
            risk_level, color, suggestion = "High Risk", "🔴", "Preventive measures and close monitoring required"

        return f"""
{color} **Prediction Result: {risk_level}**

**RAO Probability:** {prob*100:.2f}%

**Clinical Recommendation:** {suggestion}

---

*Prediction based on CatBoost machine learning model — for reference only.*
"""
    except Exception as e:
        return f"❌ Prediction failed: {str(e)}"

# ---------------------------
# 按钮
# ---------------------------
if st.button("🚀 Calculate RAO Risk"):
    st.markdown(predict_risk(Compressiontime, IntraopNTG, PreRaddiam, SRratio,
                             Heparincategory, Punctureattempts, Priorradpunctures))

if st.button("🔄 Reset"):
    st.experimental_rerun()

st.markdown("---")
st.markdown("*This tool uses machine learning for prediction. Results are for reference only and should not replace clinical judgment.*")












