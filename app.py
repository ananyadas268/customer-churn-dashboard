import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# Add any other model imports you need here, e.g., from xgboost import XGBClassifier

# -------------------------
# PAGE CONFIGURATION
# -------------------------
st.set_page_config(page_title="Customer Churn Prediction Dashboard", page_icon="📊", layout="wide")

# -------------------------
# LOAD MODELS AND FEATURES
# -------------------------
# NOTE: Ensure 'feature_names.pkl', 'logistic_model.pkl', 'random_forest_model.pkl',
# and 'xgb_model.pkl' are available in the directory.
try:
    feature_names = joblib.load('feature_names.pkl')
    logistic_model = joblib.load('logistic_model.pkl')
    rf_model = joblib.load('random_forest_model.pkl')
    xgb_model = joblib.load('xgb_model.pkl')
except FileNotFoundError:
    st.error("Error: Model or feature files not found. Please ensure 'feature_names.pkl', 'logistic_model.pkl', 'random_forest_model.pkl', 'random_forest_model.pkl', and 'xgb_model.pkl' are in the same directory.")
    st.stop()
    

# -------------------------
# SIDEBAR SETTINGS
# -------------------------
st.sidebar.title("⚙️ Settings")
advanced_mode = st.sidebar.checkbox("Enable Advanced Mode (for developer use)", value=False)

if advanced_mode:
    models = {
        "⚙️ Logistic Regression": logistic_model,
        "🌲 Random Forest": rf_model,
        "🚀 XGBoost": xgb_model
    }
    selected_model_name = st.sidebar.selectbox("Select Model", list(models.keys()))
    model = models[selected_model_name]
else:
    model = logistic_model
    selected_model_name = "⚙️ Logistic Regression (Best Model)"

st.sidebar.info(f"Model in use: {selected_model_name}")

# -------------------------
# APP TITLE AND DESCRIPTION
# -------------------------
st.title("📊 Customer Churn Prediction Dashboard")
st.write("""
Welcome to the **Customer Churn Prediction App** 👋  
This tool helps identify whether a customer is likely to leave the service based on their details.  
Just fill in the information on the left and click **Predict Churn**.
""")

# -------------------------
# USER INPUT SECTION
# -------------------------
st.sidebar.header("🧾 Enter Customer Details")

def user_input():
    # Grouping inputs for better readability
    
    st.sidebar.subheader("💰 Engagement & Charges") # Enhanced with icon
    col1, col2 = st.sidebar.columns(2)
    tenure = col1.slider("Tenure (months)", 0, 72, 10, step=1) # Added step=1
    MonthlyCharges = col2.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    TotalCharges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 500.0)
    
    st.sidebar.subheader("👨‍👩‍👧 Personal & Family") # Enhanced with icon
    col3, col4, col5 = st.sidebar.columns(3)
    gender = col3.selectbox("Gender", ["Male", "Female"])
    Partner = col4.selectbox("Partner", ["Yes", "No"])
    Dependents = col5.selectbox("Dependents", ["Yes", "No"])
    
    st.sidebar.subheader("💻 Services & Contract") # Enhanced with icon
    PhoneService = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    Contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaymentMethod = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", 
                                                         "Bank transfer (automatic)", "Credit card (automatic)"])

    data = {
        'tenure': tenure,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges,
        'gender_Male': 1 if gender == 'Male' else 0,
        'Partner_Yes': 1 if Partner == 'Yes' else 0,
        'Dependents_Yes': 1 if Dependents == 'Yes' else 0,
        'PhoneService_Yes': 1 if PhoneService == 'Yes' else 0,
        'InternetService_Fiber optic': 1 if InternetService == 'Fiber optic' else 0,
        'InternetService_No': 1 if InternetService == 'No' else 0,
        'Contract_One year': 1 if Contract == 'One year' else 0,
        'Contract_Two year': 1 if Contract == 'Two year' else 0,
        'PaymentMethod_Electronic check': 1 if PaymentMethod == 'Electronic check' else 0,
        'PaymentMethod_Mailed check': 1 if PaymentMethod == 'Mailed check' else 0,
        'PaymentMethod_Credit card (automatic)': 1 if PaymentMethod == 'Credit card (automatic)' else 0,
    }

    df = pd.DataFrame([data])
    return df

# Moved user_input function call out of the sidebar block
input_df = user_input()
input_df = input_df.reindex(columns=feature_names, fill_value=0)

# -------------------------
# MAIN SECTIONS
# -------------------------
# UPDATED: Added Tab 5 for Individual Explanation
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔮 Prediction", "📈 Model Performance", "📊 Data Insights", "🌟 Feature Importance", "💡 Individual Explanation"])

# --- TAB 1: PREDICTION (VISUAL ENHANCEMENT) ---
with tab1:
    st.header("🔮 Customer Churn Prediction")
    st.write("Review the current customer's data and click the button below to generate a real-time prediction.")

    st.subheader("Customer Input Data Summary")
    # Using st.dataframe for a nicer, scrollable table view
    st.dataframe(input_df.T.rename(columns={0: 'Input Value'}), use_container_width=True) 

    st.markdown("---") # Visual separator

    if st.button("🚀 Predict Churn Risk", type="primary", use_container_width=True):
        prediction = model.predict(input_df)[0]
        
        # Calculate probability.
        probability = "N/A"
        if hasattr(model, "predict_proba"):
            if isinstance(model, LogisticRegression):
                 probability = model.predict_proba(input_df)[:, 1][0]
            else:
                 probability = model.predict_proba(input_df)[0][1]

        # --- Visual Result Display ---
        col_result, col_prob = st.columns([2, 1])

        if prediction == 1:
            result_color = "#E53E3E" # Red 600 for contrast
            result_text = "HIGH RISK OF CHURN"
            result_emoji = "🚨"
            st.warning("Immediate action is recommended to retain this customer.")
        else:
            result_color = "#38A169" # Green 600 for contrast
            result_text = "LOW RISK (LIKELY TO STAY)"
            result_emoji = "✨"
            st.success("The customer is stable. Continue monitoring engagement.")
            
        with col_result:
             # Using custom markdown block for a strong visual prediction status
             st.markdown(f"""
                <div style='background-color: #f0f2f6; padding: 50px; border-radius: 12px; border-left: 8px solid {result_color}; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);'>
                    <h3 style='color: #000000; margin: 0;'>{result_emoji} PREDICTION STATUS</h3>
                    <h1 style='color: #000000; margin: 10px 0 0 0; font-size: 2.5em;'>{result_text}</h1>
                </div>
            """, unsafe_allow_html=True)

        with col_prob:
            if isinstance(probability, float):
                # Using st.metric for the probability display
                prob_percentage = probability * 100
                delta_val = f"{(prob_percentage - 50):.1f}% vs 50%" # How far from the neutral 50% line
                
                st.metric(
                    label="Churn Likelihood Score", 
                    value=f"{prob_percentage:.1f}%", 
                    delta=delta_val,
                    delta_color="normal" if prob_percentage > 50 else "inverse" # Red/Green delta logic
                )
            else:
                st.info("Probability not available.")

        st.balloons() # Confetti effect for a celebratory/important result!


# --- TAB 2: MODEL PERFORMANCE ---
with tab2:
    st.subheader("📊 Model Evaluation Metrics")
    model_accuracy = {
    "⚙️ Logistic Regression": "82%",
    "🌲 Random Forest": "79%",
    "🚀 XGBoost": "80%",
    "⚙️ Logistic Regression (Best Model)": "82%"  # Added to handle default mode
}

    st.info(f"Model Accuracy: **{model_accuracy.get(selected_model_name, 'N/A')}** (from training results)")

    
    cm = {
    "⚙️ Logistic Regression": np.array([[934, 102], [150, 223]]),
    "⚙️ Logistic Regression (Best Model)": np.array([[934, 102], [150, 223]]),  # Added to handle default mode
    "🌲 Random Forest": np.array([[943, 93], [200, 173]]),
    "🚀 XGBoost": np.array([[922, 114], [174, 199]])
}[selected_model_name]


    labels = ["Stayed", "Churned"]
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {selected_model_name}")
    st.pyplot(fig)

# --- TAB 3: DATA INSIGHTS ---
with tab3:
    st.subheader("📊 Churn Distribution (Example Visualization)")
    fig, ax = plt.subplots()
    churn_counts = [1036, 373]
    labels = ['Stayed', 'Churned']
    sns.barplot(x=labels, y=churn_counts, palette='Set2', ax=ax)
    ax.set_ylabel("Number of Customers")
    ax.set_title("Churn Distribution")
    st.pyplot(fig)

    st.markdown("""
    **Observations:**
    - Most customers tend to stay.
    - Around 26–30% of customers churn.
    - Model accuracy ~82%, recall for churned customers is around 60%.
    """)

# --- TAB 4: FEATURE IMPORTANCE (FIXED) ---
with tab4:
    st.subheader(f"🌟 Feature Importance - {selected_model_name}")

    try:
        # **FIXED LOGIC:** Check if the name starts with "⚙️ Logistic Regression" 
        # to cover both "⚙️ Logistic Regression" and "⚙️ Logistic Regression (Best Model)".
        if selected_model_name.startswith("⚙️ Logistic Regression"):
            # Use coefficients for Logistic Regression
            importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.coef_[0]
            }).sort_values(by='Importance', ascending=False)
        else:
            # Use feature_importances_ for tree-based models (RF, XGB)
            importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)

        top_n = st.slider("Select number of features to display", 5, 20, 10)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x='Importance', y='Feature', data=importance.head(top_n), palette='coolwarm', ax=ax)
        ax.set_title(f"Top {top_n} Important Features ({selected_model_name})")
        st.pyplot(fig)

        if selected_model_name.startswith("⚙️ Logistic Regression"):
            st.markdown("""
            **Interpretation for Logistic Regression (Coefficients):**
            - Features with **positive** values increase churn likelihood (positive correlation with the target class).
            - Features with **negative** values reduce churn likelihood (negative correlation with the target class).
            """)
        else:
             st.markdown("""
            **Interpretation for Tree-based Models (Gini/Gain Importance):**
            - Importance values show how much each feature contributed to the final prediction score. Higher is more important.
            - Importance values are always **non-negative**.
            """)

    except Exception as e:
        st.error(f"Could not display feature importance. Error details: {e}")
        st.info("Tip: This usually happens if the selected model is not loaded correctly or its type doesn't match the expected attribute (e.g., a tree model that doesn't use `feature_importances_`).")


# --- TAB 5: INDIVIDUAL EXPLANATION (NEW FEATURE) ---
with tab5:
    st.subheader("💡 Individual Prediction Explanation")
    st.write("This chart shows the features that most strongly push the current customer's prediction toward **Stay** or **Churn**.")

    try:
        if selected_model_name.startswith("⚙️ Logistic Regression"):
            # Use coefficients for LR
            raw_importance = model.coef_[0]
            # Local impact = Coefficient * Input Feature Value
            local_impact = input_df.iloc[0].values * raw_importance
            title_text = "Feature Impact (Coefficients * Input Value)"
            
            # Use the absolute value to rank, but keep the signed value for plotting
            impact_df = pd.DataFrame({
                'Feature': feature_names,
                'Impact': local_impact
            })
            
            # Sort by absolute impact to see the strongest local drivers
            impact_df['Abs_Impact'] = impact_df['Impact'].abs()
            local_importance = impact_df.sort_values(by='Abs_Impact', ascending=False).head(10)
            
            interpretation_note = "Impact sign follows coefficients: **positive** drives Churn, **negative** drives Stay."

        else:
            # For tree-based models, SHAP is better, but we can't calculate SHAP easily.
            # Fallback: Use Global Importance + show input features (less accurate local explanation)
            st.warning("Note: For tree-based models like Random Forest and XGBoost, a true local explanation (like SHAP) requires more computation. Displaying the top 10 most influential features globally for context.")
            
            raw_importance = model.feature_importances_
            local_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': raw_importance
            }).sort_values(by='Importance', ascending=False).head(10)
            
            title_text = "Top 10 Global Feature Importance (Context)"
            # Use 'Importance' column for plotting
            local_importance = local_importance.rename(columns={'Importance': 'Impact'})
            interpretation_note = "This shows global importance. Features with non-zero values are present for the customer."

        
        # --- Plotting the Local/Contextual Importance ---
        fig, ax = plt.subplots(figsize=(8, 6))
        # Use coolwarm, centered at zero, for LR coefficients
        sns.barplot(x='Impact', y='Feature', data=local_importance, palette='coolwarm', ax=ax)
        ax.set_title(title_text)
        ax.axvline(0, color='gray', linestyle='--') # Add zero line for LR
        st.pyplot(fig)
        
        st.markdown(interpretation_note)


    except Exception as e:
        st.error(f"Could not generate individual explanation: {e}")
        st.info("Ensure the selected model is fully trained and loaded correctly.")
