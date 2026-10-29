import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset

df = pd.read_csv("data/return_claims_synthetic.csv")

print("Dataset loaded:", df.shape)

# 2. Select features

numeric_features = [
"order_value_inr",
"days_since_delivery",
"num_images",
"image_blur_score",
"exif_timestamp_gap_days",
"image_reuse_flag",
"clip_image_text_sim",
"clip_image_listing_sim",
"vqa_defect_visible",
"claim_consistency_score",
"cust_total_orders_12m",
"cust_return_rate_12m",
"cust_account_age_days",
"cust_prev_fraud_flags",
"refund_requested_amount",
"delivery_pincode_risk"
]

categorical_features = [
"category",
"return_reason_code",
"complaint_language",
"payment_mode"
]

# 3. Prepare data

df["complaint_text"] = df["complaint_text"].fillna("").astype(str)

X = df[
numeric_features + categorical_features + ["complaint_text"]
]

y = df["is_fraud"]

# 4. Split dataset

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42,
stratify=y
)

# 5. Process numerical features

numeric_pipeline = Pipeline([
("imputer", SimpleImputer(strategy="median")),
("scaler", StandardScaler())
])

# 6. Process categorical features

categorical_pipeline = Pipeline([
("imputer", SimpleImputer(strategy="most_frequent")),
("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# 7. Combine features

preprocessor = ColumnTransformer([
("numeric", numeric_pipeline, numeric_features),
("categorical", categorical_pipeline, categorical_features),
(
"text",
TfidfVectorizer(max_features=3000, ngram_range=(1, 2)),
"complaint_text"
)
])

# 8. Build the model

model = Pipeline([
("preprocessor", preprocessor),
(
"classifier",
LogisticRegression(
class_weight="balanced",
max_iter=1000,
random_state=42
)
)
])

# 9. Train

print("\nTraining combined model...")
model.fit(X_train, y_train)

# 10. Evaluate

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nTest accuracy:", round(accuracy, 4))
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

# 11. Save the trained pipeline

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/combined_fraud_model.pkl")

print("\nModel saved successfully!")
print("Location: models/combined_fraud_model.pkl")
