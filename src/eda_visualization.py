import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("data/return_claims_synthetic.csv")

# ---------------------------------
# Graph 1: Fraud Distribution
# ---------------------------------

fraud_counts = df["is_fraud"].value_counts().sort_index()

plt.figure(figsize=(7, 5))

plt.bar(
    ["Genuine", "Fraudulent"],
    fraud_counts.values
)

plt.title("Genuine vs Fraudulent Returns")
plt.xlabel("Return Type")
plt.ylabel("Number of Returns")

plt.tight_layout()
plt.savefig("reports/fraud_distribution.png")
plt.show()


# ---------------------------------
# Graph 2: Fraud Rate by Category
# ---------------------------------

category_fraud = (
    df.groupby("category")["is_fraud"].mean() * 100
)

plt.figure(figsize=(9, 5))

category_fraud.sort_values().plot(kind="bar")

plt.title("Fraud Rate by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Fraud Rate (%)")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("reports/fraud_by_category.png")
plt.show()


# ---------------------------------
# Graph 3: Fraud Rate by Return Reason
# ---------------------------------

reason_fraud = (
    df.groupby("return_reason_code")["is_fraud"].mean() * 100
)

plt.figure(figsize=(10, 5))

reason_fraud.sort_values().plot(kind="bar")

plt.title("Fraud Rate by Return Reason")
plt.xlabel("Return Reason")
plt.ylabel("Fraud Rate (%)")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("reports/fraud_by_reason.png")
plt.show()

print("All three graphs were saved in the reports folder.")