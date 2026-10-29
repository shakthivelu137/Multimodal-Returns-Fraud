import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re

# Load the dataset
df = pd.read_csv("data/return_claims_synthetic.csv")

print("=" * 45)
print("CUSTOMER COMPLAINT TEXT ANALYSIS")
print("=" * 45)

# Check missing complaint descriptions
print("\n1. Missing complaint descriptions:")
print(df["complaint_text"].isnull().sum())

# Check the number of complaints
print("\n2. Total complaints:")
print(len(df["complaint_text"]))

# Display sample complaints
print("\n3. Example complaints:")

for complaint in df["complaint_text"].dropna().head(10):
    print("-", complaint)

# Calculate complaint length
df["complaint_length"] = (
    df["complaint_text"]
    .fillna("")
    .astype(str)
    .apply(len)
)

print("\n4. Complaint length statistics:")
print(df["complaint_length"].describe().round(2))

# Compare complaint lengths by fraud label
print("\n5. Average complaint length by fraud class:")

print(
    df.groupby("is_fraud")["complaint_length"]
    .mean()
    .round(2)
)

# Extract frequently used words
all_text = " ".join(
    df["complaint_text"].dropna().astype(str)
).lower()

words = re.findall(r"\b[a-z]+\b", all_text)

# Remove common words
stop_words = {
    "the", "and", "was", "is", "it", "a", "to",
    "of", "for", "in", "on", "my", "this", "with",
    "but", "not", "i", "we", "they", "very"
}

filtered_words = [
    word for word in words
    if word not in stop_words
]

word_counts = Counter(filtered_words)

print("\n6. Top 15 complaint words:")

for word, count in word_counts.most_common(15):
    print(word, ":", count)

# Plot complaint length distribution
plt.figure(figsize=(8, 5))

plt.hist(df["complaint_length"], bins=30)

plt.title("Customer Complaint Length Distribution")
plt.xlabel("Number of characters")
plt.ylabel("Number of complaints")

plt.tight_layout()

plt.savefig("reports/complaint_length_distribution.png")

plt.show()

print("\nText analysis completed!")
print("Chart saved to reports/complaint_length_distribution.png")