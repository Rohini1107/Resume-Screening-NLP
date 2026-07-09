import pandas as pd
import joblib

from preprocessing import clean_resume

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

print("Loading Dataset...")

df = pd.read_csv(
    "dataset/Resume.csv",
    encoding="utf-8",
    engine="python",
    on_bad_lines="skip"
)

df = df[['Resume_str', 'Category']]

df.rename(columns={'Resume_str': 'Resume'}, inplace=True)

print("Dataset Shape :", df.shape)

# ---------------------------------------------------
# Clean Resume
# ---------------------------------------------------

print("\nCleaning Resume Text...")

df['Clean_Resume'] = df['Resume'].apply(clean_resume)

print("Cleaning Completed!")

# ---------------------------------------------------
# Encode Target
# ---------------------------------------------------

encoder = LabelEncoder()

df['Category'] = encoder.fit_transform(df['Category'])

# ---------------------------------------------------
# TF-IDF
# ---------------------------------------------------

tfidf = TfidfVectorizer(
    max_features=20000,
    min_df=2,
    max_df=0.90,
    ngram_range=(1,2),
    stop_words='english',
    sublinear_tf=True
)

X = tfidf.fit_transform(df['Clean_Resume'])

y = df['Category']

# ---------------------------------------------------
# Train Test Split
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------
# Models
# ---------------------------------------------------

models = {
    "Linear SVM": LinearSVC(C=2, max_iter=10000),

    "Logistic Regression":
    LogisticRegression(
        max_iter=3000,
        C=2
    ),

    "Naive Bayes":
    MultinomialNB(alpha=0.1)
}

best_accuracy = 0
best_model = None
best_name = ""

print("\nTraining Models...\n")

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("="*50)
    print(name)
    print("Accuracy :", round(accuracy*100,2),"%")
    print("="*50)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_name = name

# ---------------------------------------------------
# Best Model
# ---------------------------------------------------

print("\nBest Model :", best_name)
print("Best Accuracy :", round(best_accuracy*100,2),"%")

# ---------------------------------------------------
# Final Evaluation
# ---------------------------------------------------

final_predictions = best_model.predict(X_test)

print("\nClassification Report\n")

print(classification_report(y_test, final_predictions))

# ---------------------------------------------------
# Save Model
# ---------------------------------------------------

joblib.dump(best_model, "saved_models/resume_classifier.pkl")

joblib.dump(tfidf, "saved_models/tfidf.pkl")

joblib.dump(encoder, "saved_models/label_encoder.pkl")

print("\nModel Saved Successfully!")