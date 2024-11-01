import streamlit as st
import sklearn
import joblib

st.title('Hey! Everyone')
model = joblib.load("spam_detector.h5")
vectorizer = joblib.load("vectorizer.h5")

def predict_spam(text):
    # Transform the input text with the same CountVectorizer used during training
    text_vector = vectorizer.transform([text])
    # Predict using the loaded model
    prediction = model.predict(text_vector)
    return prediction[0]

# Streamlit UI
st.title("Spam Detection System")
user_input = st.text_area("Enter text to classify as Spam or Ham")

if st.button("Predict"):
    result = predict_spam(user_input)
    if result == 1:
        st.error("This text is classified as Spam.")
    else:
        st.success("This text is classified as Ham.")