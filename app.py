# 1. Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# --- SET PAGE CONFIG ---
# This MUST be the first Streamlit command in the script
st.set_page_config(layout="wide")
# --- END PAGE CONFIG ---

# 2. Model Training Function (with caching)
# This function will only run once, and the results are cached.
@st.cache_data
def load_data():
    """Loads and preprocesses the Pima Indians Diabetes Dataset."""
    try:
        # data = pd.read_csv("https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv")
        # --- CHANGE ---
        # Load the local file instead of the URL
        data = pd.read_csv("diabetes.csv")
        # --- END CHANGE ---
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None, None, None

    # --- Data Preprocessing ---
    columns_to_clean = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    data[columns_to_clean] = data[columns_to_clean].replace(0, np.nan)
    
    for col in columns_to_clean:
        data[col] = data[col].fillna(data[col].median())
    
    # Define Features (X) and Target (y)
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    feature_names = list(X.columns)
    
    return X, y, feature_names

@st.cache_resource
def train_model(X, y):
    """Trains a Logistic Regression model and the associated scaler."""
    # Split the data
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train the Logistic Regression Model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

# --- Load Data and Train Model ---
X, y, feature_names = load_data()
if X is not None:
    model, scaler = train_model(X, y)
else:
    st.stop()


# 3. Streamlit UI
# st.set_page_config(layout="wide") # <-- This line has been moved to the top
st.title("🩺 Interactive Diabetes Risk Dashboard")
st.write("Use the controls on the left to see how changing different factors affects the diabetes risk prediction.")

# --- Define feature ranges for sliders ---
slider_ranges = {
    'Pregnancies': (0, 17, 1),
    'Glucose': (40, 200, 1),
    'BloodPressure': (20, 122, 1),
    'SkinThickness': (7, 99, 1),
    'Insulin': (14, 850, 1),
    'BMI': (18.0, 67.0, 0.1),
    'DiabetesPedigreeFunction': (0.07, 2.5, 0.01),
    'Age': (21, 81, 1)
}

# 4. Sidebar for User Inputs
st.sidebar.header("Patient Inputs")
inputs = {}
for feature, (min_val, max_val, step) in slider_ranges.items():
    # --- FIX ---
    # Ensure all slider arguments are of the same type (float) to prevent API error
    # The error "value has type list" can be misleading; it's often a type mismatch
    # between int (min/max) and float (value).
    inputs[feature] = st.sidebar.slider(
        label=feature,
        min_value=float(min_val),
        max_value=float(max_val),
        value=float(X[feature].mean()), # Default to the mean
        step=float(step)
    )

st.sidebar.header("What-If Analysis")
feature_to_analyze = st.sidebar.selectbox(
    "Analyze Feature:",
    options=feature_names,
    index=feature_names.index('Glucose') # Default to Glucose
)

# 5. Main Panel (Prediction & Plot)
col1, col2 = st.columns([1, 2])

# --- Prediction Logic ---
with col1:
    st.subheader("Risk Prediction")
    
    # Create input array in the correct order
    input_list = [inputs[feature] for feature in feature_names]
    input_data = np.array([input_list])
    
    # Scale the input
    input_scaled = scaler.transform(input_data)
    
    # Predict probability
    prediction_proba = model.predict_proba(input_scaled)[0][1]
    prediction_percent = prediction_proba * 100
    
    # Display the result with a color-coded metric
    if prediction_percent >= 50:
        st.metric(
            label="Risk Assessment",
            value=f"{prediction_percent:.2f}%",
            delta="HIGH RISK",
            delta_color="inverse"
        )
    else:
        st.metric(
            label="Risk Assessment",
            value=f"{prediction_percent:.2f}%",
            delta="LOW RISK",
            delta_color="normal"
        )
    
    st.write("This prediction is based on a Logistic Regression model. It is not a substitute for professional medical advice.")

# --- What-If Graph Logic ---
with col2:
    st.subheader(f"Risk vs. {feature_to_analyze}")

    feature_index = feature_names.index(feature_to_analyze)
    current_value = input_list[feature_index]
    
    # Get min/max from our dictionary
    feat_min, feat_max, _ = slider_ranges[feature_to_analyze]
    
    # Generate 100 values within this range
    feature_range = np.linspace(feat_min, feat_max, 100)
    
    probabilities = []
    
    # Iterate over the range
    for val in feature_range:
        temp_input = input_data.copy()
        temp_input[0, feature_index] = val
        temp_scaled = scaler.transform(temp_input)
        prob = model.predict_proba(temp_scaled)[0][1]
        probabilities.append(prob)
        
    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot the probability curve
    ax.plot(feature_range, probabilities, label="Diabetes Risk Probability", color="#0072B2", linewidth=2)
    
    # Add the "You are here" marker
    ax.axvline(x=current_value, color='red', linestyle='--', label=f'Current Value: {current_value:.2f}')
    
    # Add a marker point on the curve
    current_prob_on_curve = np.interp(current_value, feature_range, probabilities)
    ax.plot(current_value, current_prob_on_curve, 'ro', markersize=8, label="Your Position") # 'ro' = red circle
    
    # Formatting
    ax.set_xlabel(feature_to_analyze)
    ax.set_ylabel("Probability of Diabetes")
    ax.set_title(f"How {feature_to_analyze} Affects Risk")
    ax.set_ylim(0, 1) # Ensure Y-axis is from 0 to 1
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Display the plot
    st.pyplot(fig)