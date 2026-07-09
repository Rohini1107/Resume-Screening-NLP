import joblib

from preprocessing import clean_resume

# Load saved models
model = joblib.load("saved_models/resume_classifier.pkl")
tfidf = joblib.load("saved_models/tfidf.pkl")
encoder = joblib.load("saved_models/label_encoder.pkl")

print("="*60)
print("      RESUME SCREENING USING NLP")
print("="*60)

resume = input("\nPaste Resume Text:\n\n")

cleaned_resume = clean_resume(resume)

vector = tfidf.transform([cleaned_resume])

prediction = model.predict(vector)

category = encoder.inverse_transform(prediction)

print("\nPredicted Category :", category[0])