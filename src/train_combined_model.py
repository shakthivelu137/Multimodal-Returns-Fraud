import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
accuracy_score,
confusion_matrix,
classification_report
)

# 1. Load the dataset

df = pd.read_csv("data/return_claims_synthetic.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# 2. Select numerical features

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

# 3. Select categorical features

categorical_features = [
"category",
"return_reason_code",
"complaint_language",
"payment_mode"
]

# 4. Prepare the input and target

df["complaint_text"] = df["complaint_text"].fillna("").astype(str)

X = df[
numeric_features + categorical_features + ["complaint_text"]
]

y = df["is_fraud"]

# 5. Split the dataset

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42,
stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# 6. Prepare numerical features

numeric_pipeline = Pipeline([
("imputer", SimpleImputer(strategy="median")),
("scaler", StandardScaler())
])

# 7. Prepare categorical features

categorical_pipeline = Pipeline([
("imputer", SimpleImputer(strategy="most_frequent")),
("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# 8. Combine all feature-processing methods

preprocessor = ColumnTransformer([
("numeric", numeric_pipeline, numeric_features),
("categorical", categorical_pipeline, categorical_features),
(
"text",
TfidfVectorizer(max_features=3000, ngram_range=(1, 2)),
"complaint_text"
)
])

# 9. Create the complete ML pipeline

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

# 10. Train the model

print("\nTraining combined model...")

model.fit(X_train, y_train)

print("Training completed!")

# 11. Predict test results

y_pred = model.predict(X_test)

# 12. Evaluate the model

print("\nAccuracy:")
print(round(accuracy_score(y_test, y_pred), 4))

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

print("\nCombined model evaluation completed!")
