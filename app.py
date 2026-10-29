import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# Project folder & Model loading
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "combined_fraud_model.pkl"

@st.cache_resource
def load_fraud_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

model = load_fraud_model()

st.title("🛡️ Fraudulent Product Return Detection")
st.write(
    "Analyze return claims using a multimodal machine-learning model "
    "and review potentially suspicious transactions."
)

st.warning(
    "This is an operational decision support model. Predictions are not proof of fraud."
)

# Sidebar
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Predict Fraud", "CSV Analysis"]
)

# ----------------- Dashboard Page -----------------
if page == "Dashboard":
    st.header("Project Dashboard")

    if model is not None:
        st.success("✅ Machine Learning Model is active & loaded successfully.")
    else:
        st.error("❌ Model not found in models/combined_fraud_model.pkl.")

    cleaned_file = PROJECT_DIR / "data" / "cleaned_return_claims.csv"

    if cleaned_file.exists():
        df = pd.read_csv(cleaned_file)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Ingested Claims", f"{len(df):,}")

        if "is_fraud" in df.columns:
            fraud_count = int(df["is_fraud"].sum())
            col2.metric("Fraud-Labeled Claims", f"{fraud_count:,}")
            col3.metric(
                "Fraud-Labeled Percentage",
                f"{df['is_fraud'].mean() * 100:.2f}%"
            )

            st.subheader("Fraud Label Distribution")
            counts = df["is_fraud"].value_counts().sort_index()
            counts.index = [
                "Genuine (0)" if i == 0 else "Fraud (1)"
                for i in counts.index
            ]
            st.bar_chart(counts)

        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.info("Cleaned dataset not found at data/cleaned_return_claims.csv.")

# ----------------- Prediction Page -----------------
elif page == "Predict Fraud":
    st.header("Predict a Return Claim")
    st.write("Enter the claim information below to evaluate in real-time.")

    with st.form("claim_form"):
        col_a, col_b = st.columns(2)
        category = col_a.selectbox(
            "Product Category",
            ["Electronics", "Fashion", "Home", "Beauty", "Footwear", "Appliances"]
        )
        return_reason = col_b.selectbox(
            "Return Reason",
            ["Damaged", "Missing parts", "Not as described", "Quality issue", "Size issue", "Wrong item"]
        )

        col_c, col_d = st.columns(2)
        order_value = col_c.number_input("Order Value (INR)", min_value=0.0, value=2500.0)
        days_since_delivery = col_d.number_input("Days Since Delivery", min_value=0.0, value=3.0)

        col_e, col_f = st.columns(2)
        payment_mode = col_e.selectbox("Payment Mode", ["COD", "UPI", "Card", "NetBanking", "Wallet"])
        complaint_language = col_f.selectbox("Complaint Language", ["en", "hi", "ta", "hinglish"])

        complaint_text = st.text_area(
            "Customer Complaint Text",
            value="Item received was damaged and broken inside the packaging."
        )

        st.markdown("#### Multimodal & Claim-Consistency (CCS) Signals")
        m1, m2 = st.columns(2)
        clip_image_text_sim = m1.slider("CLIP Image-Text Similarity (0 = Mismatch, 1 = Perfect)", 0.0, 1.0, 0.75)
        clip_image_listing_sim = m2.slider("CLIP Catalog-Listing Match (0 = Different Item, 1 = Same)", 0.0, 1.0, 0.80)

        m3, m4 = st.columns(2)
        vqa_defect_visible = m3.selectbox("VQA Defect Visible in Photo?", [1.0, 0.0], index=0)
        image_reuse_flag = m4.selectbox("Image Reused in Previous Claims? (pHash)", [0, 1], index=0)

        m5, m6 = st.columns(2)
        claim_consistency_score = m5.slider("Claim-Consistency Score (CCS)", 0.0, 1.0, 0.82)
        delivery_pincode_risk = m6.slider("Delivery Pincode Risk", 0.0, 1.0, 0.08)

        st.markdown("#### Customer Behavioral History")
        b1, b2, b3 = st.columns(3)
        cust_total_orders = b1.number_input("Orders (Last 12m)", min_value=0, value=15)
        cust_return_rate = b2.slider("Customer Return Rate", 0.0, 1.0, 0.10)
        cust_account_age = b3.number_input("Account Age (Days)", min_value=0, value=365)

        submit = st.form_submit_button("Run Fraud Prediction", use_container_width=True)

    if submit:
        if model is None:
            st.error("Model could not be loaded.")
        else:
            payload = {
                "order_value_inr": float(order_value),
                "days_since_delivery": float(days_since_delivery),
                "num_images": 2,
                "image_blur_score": 0.15,
                "exif_timestamp_gap_days": 1.0,
                "image_reuse_flag": int(image_reuse_flag),
                "clip_image_text_sim": float(clip_image_text_sim),
                "clip_image_listing_sim": float(clip_image_listing_sim),
                "vqa_defect_visible": float(vqa_defect_visible),
                "claim_consistency_score": float(claim_consistency_score),
                "cust_total_orders_12m": int(cust_total_orders),
                "cust_return_rate_12m": float(cust_return_rate),
                "cust_account_age_days": int(cust_account_age),
                "cust_prev_fraud_flags": 0,
                "refund_requested_amount": float(order_value),
                "delivery_pincode_risk": float(delivery_pincode_risk),
                "category": category,
                "return_reason_code": return_reason,
                "complaint_language": complaint_language,
                "payment_mode": payment_mode,
                "complaint_text": complaint_text
            }

            input_data = pd.DataFrame([payload])
            prediction = int(model.predict(input_data)[0])
            probabilities = model.predict_proba(input_data)[0]
            fraud_index = list(model.classes_).index(1)
            fraud_probability = float(probabilities[fraud_index])

            st.divider()
            st.subheader("Prediction Result")
            res_col1, res_col2 = st.columns(2)

            if prediction == 1:
                res_col1.error("🚨 Classification: **Potentially Fraudulent / Abusive**")
            else:
                res_col1.success("✅ Classification: **Likely Genuine Return**")

            res_col2.metric("Fraud Probability", f"{fraud_probability * 100:.2f}%")

            if image_reuse_flag == 1:
                st.warning("⚠️ Critical Signal: Photo hash matched a previously submitted return claim.")
            if vqa_defect_visible == 0:
                st.info("ℹ️ Visual inspection signal: Claimed defect not visibly verified in image.")

# ----------------- CSV Analysis Page -----------------
elif page == "CSV Analysis":
    st.header("Dataset Analysis & Visualizations")
    cleaned_file = PROJECT_DIR / "data" / "cleaned_return_claims.csv"

    if cleaned_file.exists():
        df = pd.read_csv(cleaned_file)
        st.write(f"Displaying statistics for **{len(df):,}** return records.")

        tab1, tab2 = st.tabs(["Category Analysis", "Return Reason Breakdown"])
        with tab1:
            if "category" in df.columns and "is_fraud" in df.columns:
                cat_summary = df.groupby("category")["is_fraud"].agg(["count", "mean"]).reset_index()
                cat_summary["Fraud Rate %"] = (cat_summary["mean"] * 100).round(2)
                st.dataframe(cat_summary[["category", "count", "Fraud Rate %"]], use_container_width=True)
                st.bar_chart(data=cat_summary, x="category", y="Fraud Rate %")
        with tab2:
            if "return_reason_code" in df.columns and "is_fraud" in df.columns:
                reason_summary = df.groupby("return_reason_code")["is_fraud"].agg(["count", "mean"]).reset_index()
                reason_summary["Fraud Rate %"] = (reason_summary["mean"] * 100).round(2)
                st.dataframe(reason_summary[["return_reason_code", "count", "Fraud Rate %"]], use_container_width=True)
                st.bar_chart(data=reason_summary, x="return_reason_code", y="Fraud Rate %")
    else:
        st.info("Dataset file not found.")
