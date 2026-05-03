import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="GenoRiskBench",
    page_icon="🧬",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    
    .main {
        background-color: #0a0a0f;
        color: #e8e8f0;
    }
    
    .stApp {
        background-color: #0a0a0f;
    }
    
    h1, h2, h3 {
        font-family: 'IBM Plex Mono', monospace !important;
        color: #00ff9d !important;
    }
    
    .metric-card {
        background: #12121e;
        border: 1px solid #1e1e3a;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        margin: 5px;
    }
    
    .metric-card h3 {
        font-size: 14px !important;
        color: #888 !important;
        margin-bottom: 8px;
        font-family: 'IBM Plex Mono', monospace !important;
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: 600;
        font-family: 'IBM Plex Mono', monospace;
    }
    
    .yes-prediction {
        color: #ff4d6d;
    }
    
    .no-prediction {
        color: #00ff9d;
    }
    
    .confidence {
        font-size: 13px;
        color: #666;
        margin-top: 4px;
        font-family: 'IBM Plex Mono', monospace;
    }
    
    .header-section {
        background: linear-gradient(135deg, #0a0a0f 0%, #12122a 100%);
        border-bottom: 1px solid #1e1e3a;
        padding: 20px 0;
        margin-bottom: 30px;
    }
    
    .stSlider label {
        color: #aaa !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 13px !important;
    }
    
    .sidebar-title {
        font-family: 'IBM Plex Mono', monospace;
        color: #00ff9d;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 15px;
    }
    
    div[data-testid="stSidebar"] {
        background-color: #0d0d1a;
        border-right: 1px solid #1e1e3a;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #00ff9d, #00b4d8);
        color: #0a0a0f;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        border: none;
        border-radius: 4px;
        padding: 12px 30px;
        font-size: 14px;
        letter-spacing: 1px;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #00b4d8, #00ff9d);
        transform: translateY(-1px);
    }
    
    .result-banner {
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        margin: 20px 0;
        font-family: 'IBM Plex Mono', monospace;
    }
    
    .result-positive {
        background: rgba(255, 77, 109, 0.1);
        border: 1px solid #ff4d6d;
        color: #ff4d6d;
    }
    
    .result-negative {
        background: rgba(0, 255, 157, 0.1);
        border: 1px solid #00ff9d;
        color: #00ff9d;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_and_train():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    df = pd.read_csv(url)
    
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_cols:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    
    models = {
        'Naive Bayes':   GaussianNB(),
        'SVM':           SVC(kernel='rbf', probability=True, random_state=42),
        'KNN':           KNeighborsClassifier(n_neighbors=5),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    for name, model in models.items():
        model.fit(X_train_sc, y_train)
    
    explainer = shap.TreeExplainer(models['Random Forest'])
    
    return models, scaler, explainer, X.columns.tolist()


st.markdown("""
<div style="padding: 10px 0 30px 0;">
    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #444; letter-spacing: 3px; text-transform: uppercase; margin-bottom: 8px;">
        BAHRIA UNIVERSITY · H-11 CAMPUS
    </div>
    <h1 style="font-size: 36px; margin: 0; letter-spacing: -1px;">🧬 GenoRiskBench</h1>
    <p style="color: #666; font-family: 'IBM Plex Mono', monospace; font-size: 13px; margin-top: 8px;">
        Clinical Diabetes Risk Prediction · ML vs LLM Benchmark
    </p>
</div>
""", unsafe_allow_html=True)

with st.spinner("Loading models..."):
    models, scaler, explainer, feature_names = load_and_train()

st.sidebar.markdown('<div class="sidebar-title">Patient Parameters</div>', unsafe_allow_html=True)
st.sidebar.markdown("---")

pregnancies = st.sidebar.slider("Pregnancies", 0, 17, 3)
glucose = st.sidebar.slider("Glucose (mg/dL)", 44, 199, 117)
blood_pressure = st.sidebar.slider("Blood Pressure (mm Hg)", 24, 122, 72)
skin_thickness = st.sidebar.slider("Skin Thickness (mm)", 7, 99, 23)
insulin = st.sidebar.slider("Insulin (μU/mL)", 14, 846, 80)
bmi = st.sidebar.slider("BMI", 18.0, 67.0, 32.0, step=0.1)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.08, 2.42, 0.47, step=0.01)
age = st.sidebar.slider("Age", 21, 81, 33)

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🔬 PREDICT DIABETES RISK")

input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                         insulin, bmi, dpf, age]])
input_scaled = scaler.transform(input_data)

if predict_btn:
    st.markdown("## Prediction Results")
    
    cols = st.columns(4)
    colors = {'Naive Bayes': '#534AB7', 'SVM': '#1D9E75', 
               'KNN': '#BA7517', 'Random Forest': '#D85A30'}
    
    votes_yes = 0
    for idx, (name, model) in enumerate(models.items()):
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]
        if pred == 1:
            votes_yes += 1
        
        verdict = "DIABETES" if pred == 1 else "NO DIABETES"
        css_class = "yes-prediction" if pred == 1 else "no-prediction"
        
        with cols[idx]:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{name}</h3>
                <div class="metric-value {css_class}">{verdict}</div>
                <div class="confidence">Confidence: {prob*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if votes_yes >= 3:
        st.markdown(f"""
        <div class="result-banner result-positive">
            <div style="font-size: 24px; font-weight: 600;">⚠️ HIGH DIABETES RISK</div>
            <div style="font-size: 14px; margin-top: 8px; opacity: 0.8;">{votes_yes}/4 models predict diabetes · Consult a healthcare professional</div>
        </div>
        """, unsafe_allow_html=True)
    elif votes_yes >= 2:
        st.markdown(f"""
        <div class="result-banner" style="background: rgba(255,165,0,0.1); border: 1px solid orange; color: orange; padding: 20px; border-radius: 8px; text-align: center; font-family: 'IBM Plex Mono', monospace;">
            <div style="font-size: 24px; font-weight: 600;">⚡ MODERATE RISK</div>
            <div style="font-size: 14px; margin-top: 8px; opacity: 0.8;">{votes_yes}/4 models predict diabetes · Consider medical consultation</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-banner result-negative">
            <div style="font-size: 24px; font-weight: 600;">✅ LOW DIABETES RISK</div>
            <div style="font-size: 14px; margin-top: 8px; opacity: 0.8;">{votes_yes}/4 models predict diabetes · Maintain healthy lifestyle</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("## SHAP Explainability — What's Driving This Prediction?")
    st.markdown("<p style='color: #666; font-size: 13px;'>Based on Random Forest model analysis</p>", 
                unsafe_allow_html=True)
    
    shap_values = explainer.shap_values(input_scaled)
    
    if len(np.array(shap_values).shape) == 3:
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]
    
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Value': sv,
        'Input Value': input_data[0]
    }).sort_values('SHAP Value', key=abs, ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#12121e')
    ax.set_facecolor('#12121e')
    
    colors_shap = ['#ff4d6d' if v > 0 else '#00ff9d' for v in feature_importance['SHAP Value']]
    bars = ax.barh(feature_importance['Feature'], 
                   feature_importance['SHAP Value'],
                   color=colors_shap, alpha=0.85)
    
    ax.axvline(x=0, color='#444', linewidth=0.8)
    ax.set_xlabel('SHAP Value (impact on diabetes prediction)', 
                  color='#888', fontsize=11)
    ax.tick_params(colors='#aaa', labelsize=10)
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for i, (val, feat, inp) in enumerate(zip(
        feature_importance['SHAP Value'],
        feature_importance['Feature'],
        feature_importance['Input Value']
    )):
        ax.text(val + (0.002 if val >= 0 else -0.002), i,
                f'  {inp:.1f}', va='center', ha='left' if val >= 0 else 'right',
                color='#666', fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.markdown("""
    <div style="background: #12121e; border: 1px solid #1e1e3a; border-radius: 8px; padding: 15px; margin-top: 10px;">
        <p style="color: #888; font-size: 12px; font-family: 'IBM Plex Mono', monospace; margin: 0;">
            🔴 Red bars push TOWARD diabetes prediction &nbsp;&nbsp; 
            🟢 Green bars push AWAY from diabetes prediction
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align: center; padding: 60px 20px; background: #12121e; border: 1px solid #1e1e3a; border-radius: 12px; margin: 20px 0;">
        <div style="font-size: 48px; margin-bottom: 16px;">🔬</div>
        <h3 style="color: #444 !important; font-family: 'IBM Plex Mono', monospace; font-size: 16px;">
            Adjust patient parameters and click PREDICT
        </h3>
        <p style="color: #333; font-size: 13px; font-family: 'IBM Plex Mono', monospace;">
            4 ML models · SHAP explainability · Clinical diabetes risk assessment
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="display: flex; justify-content: space-between; padding: 10px 0;">
    <div style="text-align: center; flex: 1;">
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 22px; color: #00ff9d; font-weight: 600;">12</div>
        <div style="font-size: 11px; color: #555; font-family: 'IBM Plex Mono', monospace;">Models Benchmarked</div>
    </div>
    <div style="text-align: center; flex: 1;">
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 22px; color: #00ff9d; font-weight: 600;">768</div>
        <div style="font-size: 11px; color: #555; font-family: 'IBM Plex Mono', monospace;">Patients Analyzed</div>
    </div>
    <div style="text-align: center; flex: 1;">
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 22px; color: #00ff9d; font-weight: 600;">81.79%</div>
        <div style="font-size: 11px; color: #555; font-family: 'IBM Plex Mono', monospace;">Best AUC-ROC</div>
    </div>
    <div style="text-align: center; flex: 1;">
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 22px; color: #00ff9d; font-weight: 600;">SHAP</div>
        <div style="font-size: 11px; color: #555; font-family: 'IBM Plex Mono', monospace;">Explainability</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; padding: 20px 0; color: #333; font-family: 'IBM Plex Mono', monospace; font-size: 11px;">
    GenoRiskBench · Rahaf Tanveer & Alishba Zulfiqar · Bahria University H-11 · 2025
</div>
""", unsafe_allow_html=True)
