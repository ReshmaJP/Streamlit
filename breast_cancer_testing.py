import streamlit as st
import tensorflow as tf
import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set dark theme for Streamlit
st.markdown(
    """
    <style>
    .reportview-container {
        background: #1E1E1E;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load Breast Cancer dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build and compile the model
N, D = X_train.shape
model = tf.keras.models.Sequential([tf.keras.layers.Input(shape=(D,)), tf.keras.layers.Dense(1, activation='sigmoid')])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
r = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100)

# Streamlit App
st.title("Breast Cancer Classification using Deep Learning")

# Display the dataset
st.subheader("Breast Cancer Dataset")
st.dataframe(df)

# Display model training results
st.subheader("Model Training Results")
st.line_chart(pd.DataFrame(r.history))

# Input Section
st.subheader("Prediction Section")

# Example input fields (customize based on your features)
input_features = []

for feature in data.feature_names:
    min_val = float(df[feature].min())
    max_val = float(df[feature].max())
    default_val = float(df[feature].mean())
    input_val = st.slider(f"{feature} Input", min_val, max_val, default_val)
    input_features.append(input_val)

# Make Prediction
prediction_button = st.button("Make Prediction")

if prediction_button:
    # Format the input data for prediction
    input_data = scaler.transform([input_features])
    
    # Make prediction
    prediction = model.predict(input_data)[0, 0]
    
    # Display the prediction result
    result_text = "The Tumor is Classified as Malignant (Cancerous)" if prediction > 0.5 else "The Tumor is Classified as Benign (Not Cancerous)"
    st.subheader("Prediction Result:")
    st.write(result_text)

# Plot accuracy and loss over epochs
st.subheader("Accuracy and Loss Over Epochs")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
ax1.plot(r.history['accuracy'], label='Train Accuracy')
ax1.plot(r.history['val_accuracy'], label='Validation Accuracy')
ax1.set_title('Model Accuracy')
ax1.legend()

ax2.plot(r.history['loss'], label='Train Loss')
ax2.plot(r.history['val_loss'], label='Validation Loss')
ax2.set_title('Model Loss')
ax2.legend()

# Display the plots in Streamlit
st.pyplot(fig)
