import streamlit as st
from model import predict_email

st.set_page_config(page_title="Email Classification", layout="centered")
st.title("📧 Email Classification System (Spam / Ham)")
st.write("Enter the content of an email to check whether it's spam or a normal (ham) email.")
st.markdown("---")

email_input = st.text_area(
    "Email Content:",
    height=200,
    placeholder="Type or paste your email content here...")

if st.button("Predict", use_container_width=True):
    if email_input and email_input.strip():
        result_num = predict_email(email_input)
        if result_num == 1:
            st.error("⚠️ **Prediction:** Spam Email (**SPAM**)")
            st.markdown("This email shows characteristics of spam and may be unwanted.")
        else:
            st.success("✅ **Prediction:** Normal Email (**HAM**)")
            st.markdown("This email appears to be a normal, non-spam message.")
    else:
        st.warning("Please enter an email message to classify.")
