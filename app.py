
import streamlit as st
import pandas as pd
import requests
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# API address
API_URL = "http://127.0.0.1:8000"

# Project folder
PROJECT_DIR = Path(__file__).resolve().parent

st.title("🛡️ Fraudulent Product Return Detection")
st.write(
    "Analyze return claims using a machine-learning model "
    "and review potentially suspicious transactions."
)

st.warning(
    "This is an experimental model. Predictions are not proof of fraud."
)

# Sidebar
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Predict Fraud", "CSV Analysis"]
)

# Dashboard page
if page == "Dashboard":
    st.header("Project Dashboard")

    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=5
        )

        if response.ok:
            st.success("API server is connected.")
        else:
            st.error("API returned an error.")

    except requests.RequestException:
        st.error(
            "API is not reachable. Start your FastAPI server first."
        )

    cleaned_file = PROJECT_DIR / "data" / "cleaned_return_claims.csv"

    if cleaned_file.exists():
        df = pd.read_csv(cleaned_file)

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Claims", f"{len(df):,}")

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
        st.info(
            "Cleaned dataset not found at "
            "data/cleaned_return_claims.csv. "
            "You can still test the API under Predict Fraud."
        )

# Prediction page
elif page == "Predict Fraud":
    st.header("Predict a Return Claim")

    st.write(
        "Enter the claim information below. "
        "The values will be sent to your existing API."
    )

    with st.form("claim_form"):
        category = st.selectbox(
            "Product Category",
            ["Electronics", "Fashion", "Home",
             "Beauty", "Footwear", "Appliances"]
        )

        return_reason = st.selectbox(
            "Return Reason",
            ["Damaged", "Missing parts", "Not as described",
             "Quality issue", "Size issue", "Wrong item"]
        )

        order_value = st.number_input(
            "Order Value (INR)",
            min_value=0.0,
            value=1000.0
        )

        days_since_delivery = st.number_input(
            "Days Since Delivery",
            min_value=0,
            value=3
        )

        num_images = st.number_input(
            "Number of Images",
            min_value=0,
            value=0
        )

        image_blur_score = st.slider(
            "Image Blur Score",
            min_value=0.0,
            max_value=1.0,
            value=0.0
        )

        exif_gap = st.number_input(
            "EXIF Timestamp Gap (Days)",
            min_value=0,
            value=0
        )

        image_reuse_flag = st.selectbox(
            "Image Reuse Flag",
            [0, 1]
        )

        clip_text_sim = st.slider(
            "Image-Text Similarity",
            0.0, 1.0, 0.7
        )

        clip_listing_sim = st.slider(
            "Image-Listing Similarity",
            0.0, 1.0, 0.7
        )

        vqa_defect_visible = st.selectbox(
            "VQA Defect Visible",
            [0, 1]
        )

        claim_consistency_score = st.slider(
            "Claim Consistency Score",
            0.0, 1.0, 0.8
        )

        cust_total_orders = st.number_input(
            "Customer Orders in Last 12 Months",
            min_value=0,
            value=5
        )

        cust_return_rate = st.slider(
            "Customer Return Rate",
            0.0, 1.0, 0.1
        )

        cust_account_age = st.number_input(
            "Customer Account Age (Days)",
            min_value=0,
            value=365
        )

        cust_prev_fraud_flags = st.number_input(
            "Previous Fraud Flags",
            min_value=0,
            value=0
        )

        refund_amount = st.number_input(
            "Refund Requested Amount (INR)",
            min_value=0.0,
            value=1000.0
        )

        delivery_pincode_risk = st.slider(
            "Delivery PIN Code Risk",
            0.0, 1.0, 0.1
        )

        payment_mode = st.selectbox(
            "Payment Mode",
            ["UPI", "Card", "Cash on Delivery", "Net Banking"]
        )

        complaint_language = st.selectbox(
            "Complaint Language",
            ["English", "Tamil", "Hindi", "Other"]
        )

        complaint_text = st.text_area(
            "Customer Complaint",
            "Product received in good condition."
        )

        submitted = st.form_submit_button("Analyze Claim")

    if submitted:
        # These names must match the ReturnClaim schema in api/main.py.
        payload = {
            "category": category,
            "return_reason_code": return_reason,
            "order_value_inr": order_value,
            "days_since_delivery": days_since_delivery,
            "num_images": num_images,
            "image_blur_score": image_blur_score,
            "exif_timestamp_gap_days": exif_gap,
            "image_reuse_flag": image_reuse_flag,
            "clip_image_text_sim": clip_text_sim,
            "clip_image_listing_sim": clip_listing_sim,
            "vqa_defect_visible": vqa_defect_visible,
            "claim_consistency_score": claim_consistency_score,
            "cust_total_orders_12m": cust_total_orders,
            "cust_return_rate_12m": cust_return_rate,
            "cust_account_age_days": cust_account_age,
            "cust_prev_fraud_flags": cust_prev_fraud_flags,
            "refund_requested_amount": refund_amount,
            "delivery_pincode_risk": delivery_pincode_risk,
            "payment_mode": payment_mode,
            "complaint_language": complaint_language,
            "complaint_text": complaint_text
        }

        try:
            result = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=30
            )

            if result.ok:
                data = result.json()

                st.subheader("Prediction Result")
                st.json(data)

                if data.get("prediction") == 1:
                    st.error("Model flagged this claim as potentially fraudulent.")
                elif data.get("prediction") == 0:
                    st.success("Model classified this claim as likely genuine.")

                if "fraud_probability_percent" in data:
                    st.metric(
                        "Reported Fraud Probability",
                        f"{data['fraud_probability_percent']}%"
                    )

                st.caption(
                    "Experimental prediction only. Review the claim "
                    "before making any decision."
                )

            else:
                st.error(f"API error: {result.status_code}")
                st.code(result.text)

        except requests.RequestException as error:
            st.error(f"Could not connect to the API: {error}")

# CSV analysis page
elif page == "CSV Analysis":
    st.header("Analyze a CSV Dataset")

    uploaded_file = st.file_uploader(
        "Upload your return claims CSV",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            st.success("CSV loaded successfully.")

            col1, col2 = st.columns(2)
            col1.metric("Rows", f"{len(df):,}")
            col2.metric("Columns", f"{len(df.columns):,}")

            st.subheader("Data Preview")
            st.dataframe(df.head(20), use_container_width=True)

            st.subheader("Missing Values")
            missing = df.isnull().sum()
            missing = missing[missing > 0]

            if missing.empty:
                st.write("No missing values found.")
            else:
                st.dataframe(
                    missing.rename("Missing Count"),
                    use_container_width=True
                )

            if "is_fraud" in df.columns:
                st.subheader("Fraud Label Distribution")
                st.bar_chart(df["is_fraud"].value_counts())

        except Exception as error:
            st.error(f"Could not read the CSV: {error}")
