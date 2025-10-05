import streamlit as st
import numpy as np
from PIL import Image
import time
import io

# Configuration for the Streamlit page
st.set_page_config(
    page_title="Integrated Oral Cancer Screening Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. Risk Factor & Symptom Assessment Logic (from previous file) ---

def calculate_risk(inputs):
    """Calculates a numerical risk score based on user inputs."""
    score = 0
    # Risk Factors
    packs_per_day = inputs['packs_per_day']
    if packs_per_day > 0: score += 10 if packs_per_day > 10 else 5
    drinks_per_week = inputs['drinks_per_week']
    if drinks_per_week > 0: score += 10 if drinks_per_week > 14 else 5
    age = inputs['age']
    if age >= 65: score += 10
    elif age >= 45: score += 5
    if inputs['family_history']: score += 15 # Significant factor

    # Symptoms/Signs (Visual Inspection) - High weights for persistent changes
    if inputs['persistent_sore']: score += 30
    if inputs['patch_changes']: score += 25
    if inputs['lump_swelling']: score += 20
    if inputs['hoarseness']: score += 10
    if inputs['swallowing_difficulty']: score += 10

    return score

def assess_risk_level(score):
    """Determines the risk category and recommended action."""
    if score >= 50:
        return {
            'level': 'High Risk',
            'color': 'red',
            'icon': '🚨',
            'action': "Immediate consultation with an Oral & Maxillofacial Surgeon or an ENT specialist is highly recommended."
        }
    elif score >= 10:
        return {
            'level': 'Moderate Risk',
            'color': 'orange',
            'icon': '⚠️',
            'action': "Schedule a comprehensive dental check-up with a professional oral cancer screening. Monitor symptoms closely."
        }
    else:
        return {
            'level': 'Low Risk',
            'color': 'green',
            'icon': '✅',
            'action': "Continue with regular dental check-ups (every 6 months) and maintain healthy lifestyle habits. Be aware of any persistent changes."
        }

# --- 2. Image Prediction Mock Function (based on Oral-Cancer.ipynb) ---

def mock_predict_image(uploaded_file):
    """
    Mocks a prediction from the VGG19 based CNN model described in Oral-Cancer.ipynb.
    
    In a real application, this function would load a saved Keras/TensorFlow model
    and process the image before making a prediction.
    """
    if uploaded_file is None:
        return None, None

    # Simulate image loading and preprocessing
    try:
        image = Image.open(uploaded_file).convert('RGB')
        
        # Simulate model running time
        with st.spinner('Analyzing image with CNN model...'):
            time.sleep(2) # Simulate network/processing delay

        # Mock results based on the likelihood of a suspicious area being uploaded
        # A simple heuristic based on image size/properties could be used, but
        # here we return a plausible mock result for demonstration.
        np.random.seed(int(time.time() * 1000) % 1000)
        
        # Two classes: 0 (Benign/Normal) and 1 (Suspicious/Malignant)
        
        # Mock probabilities (e.g., 80% confident it's Suspicious)
        if np.random.rand() < 0.7:
             # Higher chance of being suspicious if user actively uploaded an image
            probabilities = [0.2, 0.8] # [Benign, Suspicious]
        else:
            probabilities = [0.95, 0.05]
            
        prediction_index = np.argmax(probabilities)
        
        class_names = ["Benign/Normal Tissue", "Suspicious/Malignant Lesion"]
        
        result = {
            'class': class_names[prediction_index],
            'confidence': probabilities[prediction_index]
        }
        
        return image, result

    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None

# --- Main Application Logic ---

def app():
    """Renders the Streamlit application."""
    st.title("Oral Cancer Screening Tool 🗣️***Developed by vishal***")
    st.markdown("""
    This integrated tool combines lifestyle and symptom risk assessment with a preliminary image analysis.
    **Disclaimer:** This is an informational tool and cannot replace a professional medical diagnosis.
    """)

    tab1, tab2 = st.tabs(["📝 Risk Factor Assessment", "📸 Image Analysis (Experimental)"])

    # --- Tab 1: Risk Factor Assessment ---
    with tab1:
        st.header("Risk Factor & Symptom Assessment")

        # Step 1: Personal Risk Factors
        with st.expander("📝 STEP 1: Lifestyle and Demographic Factors", expanded=True):
            col1, col2, col3 = st.columns(3)

            with col1:
                age = st.slider("1. Your Age", 18, 90, 45, key="age_risk", help="Risk increases significantly after age 45.")

            with col2:
                packs_per_day = st.number_input("2. Cigarettes/Cigars Per Day (0 if non-smoker)",
                                                min_value=0, max_value=50, value=0, key="packs_risk")

            with col3:
                drinks_per_week = st.number_input("3. Alcoholic Drinks Per Week (0 if non-drinker)",
                                                  min_value=0, max_value=40, value=0, key="drinks_risk", help="1 drink = 12oz beer, 5oz wine, or 1.5oz spirit.")

            family_history = st.checkbox("4. Do you have a family history of oral, head, or neck cancer?",
                                         key="family_risk", help="Immediate family members (parent, sibling).")

        st.markdown("---")

        # Step 2: Symptoms
        with st.expander("🔍 STEP 2: Current Symptoms & Oral Inspection (Last 3 Weeks)", expanded=True):
            st.markdown("Please check any persistent signs **lasting for more than 3 weeks**.")

            col4, col5 = st.columns(2)

            with col4:
                persistent_sore = st.checkbox("5. A mouth sore or ulcer that **will not heal**.", key="sore_check")
                lump_swelling = st.checkbox("7. A lump, thickening, or swelling in your cheek, throat, or neck.", key="lump_check")
                hoarseness = st.checkbox("9. Chronic hoarseness or a change in your voice.", key="hoarseness_check")

            with col5:
                patch_changes = st.checkbox("6. Red, white, or mixed patches (erythroplakia/leukoplakia) in your mouth or on your lips.", key="patch_check")
                swallowing_difficulty = st.checkbox("8. Difficulty chewing or swallowing (dysphagia).", key="swallowing_check")
                # Removed pain_numbness from weight calculation, but kept for user reporting
                st.checkbox("10. Persistent numbness or unexplained pain in the mouth or face.", key="numbness_check")


        # --- Calculate Risk Button (Tab 1) ---
        st.markdown("---")
        if st.button("Calculate Risk Score", use_container_width=True, type="primary"):
            user_inputs = {
                'age': age, 'packs_per_day': packs_per_day, 'drinks_per_week': drinks_per_week,
                'family_history': family_history, 'persistent_sore': persistent_sore,
                'patch_changes': patch_changes, 'lump_swelling': lump_swelling,
                'hoarseness': hoarseness, 'swallowing_difficulty': swallowing_difficulty,
            }

            score = calculate_risk(user_inputs)
            assessment = assess_risk_level(score)

            # Display Results
            st.subheader(f"{assessment['icon']} Your Calculated Risk Assessment")
            st.markdown(f"**Total Risk Score:** `{score}`")
            st.markdown(f"""
            <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid {assessment['color']};'>
                <h3 style='color: {assessment['color']}; margin-top: 0px;'>{assessment['level']}</h3>
                <p style='font-size: 1.1em;'>**Recommended Action:** {assessment['action']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("Results are based on combined risk factors and symptoms. See the Image Analysis tab for another layer of screening.")

    # --- Tab 2: Image Analysis ---
    with tab2:
        st.header("Upload Oral Lesion Image for Pre-Screening")
        st.markdown("""
        Upload a clear, well-lit image of any suspicious lesion (sore, lump, patch) for preliminary analysis by a simulated **VGG19-based CNN**.
        This feature is experimental and results are mock predictions based on the architecture in your provided notebook.
        """)

        uploaded_file = st.file_uploader("Choose an image (JPEG, PNG)", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            
            st.success("Image uploaded successfully!")
            
            col_img, col_pred = st.columns([1, 2])
            
            with col_img:
                image, prediction = mock_predict_image(uploaded_file)
                if image:
                    st.image(image, caption='Image for Analysis', use_column_width=True)

            with col_pred:
                if prediction:
                    st.subheader("Model Analysis Results")
                    
                    if "Suspicious" in prediction['class']:
                        st.error(f"Prediction: {prediction['class']}")
                        st.markdown(f"Confidence: **{prediction['confidence']:.2%}**")
                        st.warning("""
                        **URGENT WARNING:** The model detected features highly consistent with a suspicious lesion.
                        **DO NOT DELAY** seeking an immediate professional evaluation.
                        """)
                    else:
                        st.success(f"Prediction: {prediction['class']}")
                        st.markdown(f"Confidence: **{prediction['confidence']:.2%}**")
                        st.info("""
                        **Preliminary Finding:** The model classified the image as generally benign or normal tissue.
                        However, if you have persistent symptoms (Tab 1), always consult a doctor.
                        """)
                    
                    st.markdown("---")
                    st.markdown("""
                    <small>The model in this app is simulated to demonstrate the potential of computer vision in screening. 
                    A real model would be trained on thousands of labeled biopsy-confirmed images to achieve clinical utility.</small>
                    """, unsafe_allow_html=True)

# Run the app function
if __name__ == "__main__":
    app()

