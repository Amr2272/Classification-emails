import joblib
from utility import clean_text
VECTORIZER = joblib.load('vectorizer.pkl')
MODEL = joblib.load('model.pkl')

def predict_email(email_text):
    cleaned_text = clean_text(email_text)
    text_vectorized = VECTORIZER.transform([cleaned_text])
    prediction = MODEL.predict(text_vectorized)[0]
    return int(prediction)