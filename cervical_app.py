import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
MODEL_PATH = 'cc_dt_model.pkl'
SCALER_PATH = 'cc_scaler.pkl'
TARGET_COLUMN = 'Biopsy'
RANDOM_STATE = 12

FEATURE_COLUMNS = [
    'Age', 'Number of sexual partners', 'First sexual intercourse',
    'Num of pregnancies', 'Smokes', 'Smokes (years)', 'Smokes (packs/year)',
    'Hormonal Contraceptives', 'Hormonal Contraceptives (years)', 'IUD',
    'IUD (years)', 'STDs', 'STDs (number)', 'STDs:condylomatosis',
    'STDs:cervical condylomatosis', 'STDs:vaginal condylomatosis',
    'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis',
    'STDs:pelvic inflammatory disease', 'STDs:genital herpes',
    'STDs:molluscum contagiosum', 'STDs:AIDS', 'STDs:HIV', 'STDs:Hepatitis B',
    'STDs:HPV', 'Hinselmann', 'Schiller', 'Cytology', 'Dx:Cancer', 'Dx:CIN',
    'Dx:HPV'
]

# --- Helper Functions ---

@st.cache_resource
def train_and_save_model():
    st.info("Training model with mock dataset...")

    # 1. GENERATE MOCK DATA
    n_samples = 850
    data = {col: np.random.rand(n_samples) for col in FEATURE_COLUMNS}
    data[TARGET_COLUMN] = np.random.randint(0, 2, n_samples)
    df = pd.DataFrame(data)

    # Add some NaNs for testing imputation
    for col in FEATURE_COLUMNS[:10]:
        df.loc[np.random.choice(n_samples, 50, replace=False), col] = np.nan

    df = df.replace('?', np.nan)
    df = df.apply(pd.to_numeric, errors='coerce')
    df.columns = df.columns.str.strip()  # remove spaces

    # 2. IMPUTATION
    imputation_map = {
        'Number of sexual partners': df['Number of sexual partners'].median(),
        'First sexual intercourse': df['First sexual intercourse'].median(),
        'Num of pregnancies': df['Num of pregnancies'].median(),
        'Smokes': 1,
        'Smokes (years)': df['Smokes (years)'].median(),
        'Smokes (packs/year)': df['Smokes (packs/year)'].median(),
        'Hormonal Contraceptives': 1,
        'Hormonal Contraceptives (years)': df['Hormonal Contraceptives (years)'].median(),
        'IUD': 0,
        'IUD (years)': 0,
        'STDs': 1,
        'STDs (number)': df['STDs (number)'].median(),
        'STDs:condylomatosis': df['STDs:condylomatosis'].median(),
        'STDs:cervical condylomatosis': df['STDs:cervical condylomatosis'].median(),
        'STDs:vaginal condylomatosis': df['STDs:vaginal condylomatosis'].median(),
        'STDs:vulvo-perineal condylomatosis': df['STDs:vulvo-perineal condylomatosis'].median(),
        'STDs:syphilis': df['STDs:syphilis'].median(),
        'STDs:pelvic inflammatory disease': df['STDs:pelvic inflammatory disease'].median(),
        'STDs:genital herpes': df['STDs:genital herpes'].median(),
        'STDs:molluscum contagiosum': df['STDs:molluscum contagiosum'].median(),
        'STDs:AIDS': df['STDs:AIDS'].median(),
        'STDs:HIV': df['STDs:HIV'].median(),
        'STDs:Hepatitis B': df['STDs:Hepatitis B'].median(),
        'STDs:HPV': df['STDs:HPV'].median(),
    }

    for col, value in imputation_map.items():
        if col in df.columns:
            df[col] = df[col].fillna(value)

    # 3. SPLIT & SCALE
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 4. TRAIN MODEL
    model = DecisionTreeClassifier(max_depth=2, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)

    # 5. SAVE
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    st.success("Model training complete!")
    return model, scaler

def load_model_and_scaler():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except FileNotFoundError:
        return train_and_save_model()


# --- Main App ---

def main():
    st.set_page_config(page_title="Cervical Cancer Risk Predictor", layout="wide")
    st.title("🔬 Cervical Cancer Risk Prediction")
    st.markdown("Enter the patient's data to predict the risk of a positive Biopsy result.")

    model, scaler = load_model_and_scaler()
    user_inputs = {}

    # Patient demographics
    col1, col2 = st.columns(2)
    with col1:
        user_inputs['Age'] = st.number_input("Age", min_value=10, max_value=100, value=30)
        user_inputs['Number of sexual partners'] = st.slider("Number of sexual partners", 1, 30, 3)
        user_inputs['First sexual intercourse'] = st.number_input("Age at first sexual intercourse", 10, 50, 17)
        user_inputs['Num of pregnancies'] = st.number_input("Number of pregnancies", 0, 20, 2)
    with col2:
        user_inputs['Smokes'] = st.selectbox("Smokes?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        user_inputs['Smokes (years)'] = st.number_input("Smokes (years)", 0.0, 50.0, 0.0)
        user_inputs['Smokes (packs/year)'] = st.number_input("Smokes (packs/year)", 0.0, 100.0, 0.0)
        user_inputs['Hormonal Contraceptives'] = st.selectbox("Uses Hormonal Contraceptives?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        user_inputs['Hormonal Contraceptives (years)'] = st.number_input("Hormonal Contraceptives (years)", 0.0, 50.0, 1.0)
        user_inputs['IUD'] = st.selectbox("Uses IUD?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        user_inputs['IUD (years)'] = st.number_input("IUD (years)", 0.0, 30.0, 0.0)

    # STDs & History
    std_cols = FEATURE_COLUMNS[11:25]
    for col in std_cols:
        user_inputs[col] = st.selectbox(col, [0, 1])

    # Prior screenings
    diag_cols = FEATURE_COLUMNS[25:]
    for col in diag_cols:
        user_inputs[col] = st.selectbox(col, [0, 1])

    # Predict
    if st.button("Predict Biopsy Result"):
        input_df = pd.DataFrame([user_inputs])[FEATURE_COLUMNS]
        scaled_data = scaler.transform(input_df)
        pred = model.predict(scaled_data)[0]

        st.divider()
        st.header("Prediction Result")
        if pred == 1:
            st.error("⚠️ HIGH RISK: POSITIVE Biopsy Predicted")
            st.markdown("_Consult a healthcare professional for diagnosis._")
        else:
            st.success("✅ LOW RISK: NEGATIVE Biopsy Predicted")
            st.markdown("_Consult a healthcare professional for diagnosis._")

if __name__ == "__main__":
    main()
