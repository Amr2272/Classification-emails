import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import SMOTE
from utility import load_and_preprocess_data
DATA_FILE_PATH = r'C:\Users\Amr\Desktop\Classfication email\Data\spam.csv'
X_cleaned, y = load_and_preprocess_data(DATA_FILE_PATH)
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y,random_state=42, stratify=y)

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), min_df=2)
X_train_vec = vectorizer.fit_transform(X_train)

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train_vec, y_train)

model = MultinomialNB()
model.fit(X_res, y_res)

joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(model, 'model.pkl')