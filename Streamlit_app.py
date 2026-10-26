import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="HR Placement Analytics & Predictor", 
    page_icon="💼", 
    layout="wide"
)


@st.cache_resource
def load_assets():
    model = joblib.load('placement_model.pkl')
    columns = joblib.load('model_columns.pkl')
    return model, columns

@st.cache_data
def load_data():
    try:
        return pd.read_csv('cleaned_HR_job_placement_dataset.csv')
    except Exception:
        return pd.read_csv('cleaned_hr_job_placement_dataset.csv')

model, model_columns = load_assets()
df_raw = load_data()


if 'avg_interview_score' not in df_raw.columns:
    df_raw['avg_interview_score'] = (
        df_raw['technical_score'] + 
        df_raw['aptitude_score'] + 
        df_raw['communication_score']
    ) / 3

st.title("💼 HR Job Acceptance & Placement Dashboard")


tab1, tab2 = st.tabs(["📊 Key KPIs & Analytics", "🔮 Single Candidate Predictor"])

# ==========================================
# TAB 1: 7 KEY KPIS (PROJECT REQUIREMENT)
# ==========================================
with tab1:
    st.subheader("Key Performance Indicators")
    
    total_candidates = len(df_raw)
    placed_count = (df_raw['status'] == 'Placed').sum()
    placement_rate = (placed_count / total_candidates) * 100
    
    if 'job_offer_accepted' in df_raw.columns:
        offers_extended = df_raw[df_raw['job_offer_accepted'].isin(['Yes', 'No', 1, 0])]
        accepted_count = (df_raw['job_offer_accepted'].isin(['Yes', 1])).sum()
        job_acceptance_rate = (accepted_count / len(offers_extended) * 100) if len(offers_extended) > 0 else placement_rate
        dropout_rate = 100 - job_acceptance_rate
    else:
        job_acceptance_rate = placement_rate
        dropout_rate = 100 - placement_rate
        
    avg_interview_score = df_raw['avg_interview_score'].mean()
    avg_skills_match = df_raw['skills_match_percentage'].mean()
    
    high_risk_candidates = df_raw[
        (df_raw['skills_match_percentage'] < 60) | (df_raw['avg_interview_score'] < 50)
    ]
    high_risk_percentage = (len(high_risk_candidates) / total_candidates) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("1. Total Candidates", f"{total_candidates:,}")
    col2.metric("2. Placement Rate", f"{placement_rate:.1f}%")
    col3.metric("3. Job Acceptance Rate", f"{job_acceptance_rate:.1f}%")
    col4.metric("4. Offer Dropout Rate", f"{dropout_rate:.1f}%")

    st.write("")
    col5, col6, col7 = st.columns(3)
    col5.metric("5. Avg Interview Score", f"{avg_interview_score:.1f} / 100")
    col6.metric("6. Avg Skills Match %", f"{avg_skills_match:.1f}%")
    col7.metric("7. High-Risk Candidates", f"{high_risk_percentage:.1f}%")

    st.write("---")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Placement Distribution")
        st.bar_chart(df_raw['status'].value_counts())
    with c2:
        st.subheader("Candidates by Company Tier")
        if 'company_tier' in df_raw.columns:
            st.bar_chart(df_raw['company_tier'].value_counts())

# ==========================================
# TAB 2: CANDIDATE PREDICTION FORM
# ==========================================
with tab2:
    st.subheader("Predict Placement for a New Candidate")
    
    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        tech_score = st.slider("Technical Score", 0, 100, 75)
        apt_score = st.slider("Aptitude Score", 0, 100, 70)
        comm_score = st.slider("Communication Score", 0, 100, 70)
        deg_percentage = st.slider("Degree Percentage", 0, 100, 75)
        skill_percentage = st.slider("Skills Match %", 0, 100, 80)
        
    with col_input2:
        exp_years = st.slider("Years of Experience", 0, 20, 2)
        prev_ctc = st.number_input("Previous CTC (in LPA)", 0.0, 40.0, 5.0, 0.5)
        exp_ctc = st.number_input("Expected CTC (in LPA)", 0.0, 50.0, 7.5, 0.5)
        comp_tier = st.selectbox("Company Tier", ["Tier 1", "Tier 2", "Tier 3"])
        internship = st.selectbox("Internship Experience", ["Yes", "No"])
        role_match = st.selectbox("Job Role Match", ["Matched", "Not Matched"])

    avg_score = (tech_score + apt_score + comm_score) / 3
    exp_cat = 'Fresher' if exp_years == 0 else ('Junior' if exp_years <= 2 else 'Senior')
    acad_band = 'Low' if deg_percentage < 65 else ('Average' if deg_percentage <= 80 else 'High')
    skill_lvl = 'Low' if skill_percentage < 60 else ('Medium' if skill_percentage <= 80 else 'High')
    eval_cat = 'Needs Improvement' if avg_score < 50 else ('Good' if avg_score <= 75 else 'Excellent')

    st.write("---")
    
    if st.button("Generate Placement Prediction", type="primary", use_container_width=True):
        candidate_data = {
            'years_of_experience': exp_years,
            'technical_score': tech_score,
            'aptitude_score': apt_score,
            'communication_score': comm_score,
            'degree_percentage': deg_percentage,
            'skills_match_percentage': skill_percentage,
            'previous_ctc_lpa': prev_ctc,
            'expected_ctc_lpa': exp_ctc,
            'company_tier': comp_tier,
            'internship_experience': internship,
            'job_role_match': role_match,
            'avg_interview_score': avg_score,
            'experience_category': exp_cat,
            'academic_band': acad_band,
            'skills_match_level': skill_lvl,
            'interview_performance_category': eval_cat
        }
        
        input_encoded = pd.get_dummies(pd.DataFrame([candidate_data]), drop_first=True)
        ready_features = input_encoded.reindex(columns=model_columns, fill_value=0)
        
        pred = model.predict(ready_features)[0]
        prob = model.predict_proba(ready_features)[0][1] * 100
        
        res1, res2 = st.columns(2)
        with res1:
            if pred == 1:
                st.success("### 🎉 Result: Placed")
            else:
                st.error("### ❌ Result: Not Placed")
        with res2:
            st.metric("Placement Confidence", f"{prob:.2f}%")