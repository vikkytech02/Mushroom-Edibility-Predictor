import streamlit as st
import pickle
import pandas as pd

# Load model and mappings
with open('mushroom_model.pkl', 'rb') as f:
    model, feature_mappings, le_target, code_to_name = pickle.load(f)

st.set_page_config(page_title="Mushroom Edibility Predictor", layout="wide")
st.title("🍄 Mushroom Edibility Predictor")
st.markdown("Select mushroom features to predict if it's **edible** or **poisonous**.")

# UI input with human-readable names
user_input = {}
cols = st.columns(3)

for i, col in enumerate(feature_mappings.keys()):
    readable_options = list(feature_mappings[col].keys())  # ['Bell', 'Flat', ...]
    with cols[i % 3]:
        user_input[col] = st.selectbox(col.replace('-', ' ').capitalize(), readable_options)

# Convert input values to numeric for prediction
input_vector = []
for col, val in user_input.items():
    mapping = feature_mappings[col]
    input_vector.append(mapping[val])  # Convert readable name to encoded number

# Predict
if st.button("Predict"):
    prediction = model.predict([input_vector])[0]
    result = le_target.inverse_transform([prediction])[0]
    
    if result == 'edible':
        st.success("✅ The mushroom is **edible**.")
    else:
        st.error("❌ The mushroom is **poisonous**.")
