# Email Classification (Spam vs Ham)

A simple email classifier trained on a labeled dataset to predict whether an email is spam or ham. The project includes a training script, a saved model pipeline, and a Streamlit app for interactive predictions.

## Project Structure

- Data/
  - spam.csv
- Scr/
  - main.py (Streamlit UI)
  - train.py (training and evaluation)
  - model.py (load model and predict)
  - utility.py (text cleaning and data loading)
- requirements.txt

## Setup

1. Create and activate a Python environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the Model

Run the training script to build the pipeline and save artifacts:

```bash
python Scr/train.py
```

This writes:
- pipeline.pkl (full pipeline)
- vectorizer.pkl (TF-IDF vectorizer)
- model.pkl (Naive Bayes model)
- metrics.json (best params and evaluation results)

## Run the App

Start the Streamlit UI:

```bash
streamlit run Scr/main.py
```

Open the provided local URL, paste an email, and click Predict.

## Notes

- The model uses TF-IDF features with SMOTE and Multinomial Naive Bayes.
- The dataset is expected at Data/spam.csv with columns: message, class (ham/spam).


