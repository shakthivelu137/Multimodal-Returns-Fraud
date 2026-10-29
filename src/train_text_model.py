import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
accuracy_score,
classification_report,
confusion_matrix
)

# 1. Load the dataset

df = pd.read_csv("data/return_claims_synthetic.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# 2. Prepare the complaint text

df["complaint_text"] = df["complaint_text"].fillna("").astype(str)

X = df["complaint_text"]
y = df["is_fraud"]

# 3. Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42,
stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# 4. Convert text into numerical features

vectorizer = TfidfVectorizer(
max_features=3000,
ngram_range=(1, 2),
sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nText converted into numerical features!")

# 5. Create the model

model = LogisticRegression(
class_weight="balanced",
max_iter=1000,
random_state=42
)

# 6. Train the model

model.fit(X_train_tfidf, y_train)

print("Text model training completed!")

# 7. Make predictions

y_pred = model.predict(X_test_tfidf)

# 8. Evaluate the model

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(
classification_report(
y_test,
y_pred,
zero_division=0
)
)
