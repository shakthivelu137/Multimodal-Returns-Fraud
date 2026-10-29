import joblib
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load cleaned dataset
df = pd.read_csv("data/cleaned_returns.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# 2. Select input features
features = [
    "order_value_inr",
    "days_since_delivery",
    "num_images",
    "image_blur_score",
    "image_reuse_flag",
    "clip_image_text_sim",
    "clip_image_listing_sim",
    "claim_consistency_score",
    "cust_total_orders_12m",
    "cust_return_rate_12m",
    "cust_account_age_days",
    "cust_prev_fraud_flags",
    "refund_requested_amount",
    "delivery_pincode_risk"
]

target = "is_fraud"

# 3. Prepare X and y
X = df[features]
y = df[target]

# 4. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# 5. Scale numeric features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Create the model
model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

# 7. Train the model
model.fit(X_train, y_train)

print("\nModel training completed!")

# 8. Make predictions
y_pred = model.predict(X_test)

# 9. Evaluate the model
print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
# Create the models folder if needed
os.makedirs("models", exist_ok=True)

# Save the trained model
joblib.dump(model, "models/fraud_model.pkl")

# Save the scaler
joblib.dump(scaler, "models/scaler.pkl")

# Save the feature names
joblib.dump(features, "models/features.pkl")

print("\nModel saved successfully!")
print("Saved files in the models folder.")