import re
import pandas as pd 
def clean_text(text):
    return re.sub(r'[^a-zA-Z\s]', '', str(text).lower())

def load_and_preprocess_data(file_path):
    try:
        df = pd.read_csv(file_path, encoding='latin-1')
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        return None, None
    
    X = df['message']
    y = df['class'].map({'ham': 0, 'spam': 1})
    X_cleaned = X.apply(clean_text)
    return X_cleaned, y