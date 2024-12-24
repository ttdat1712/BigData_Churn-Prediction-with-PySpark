import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.classification import DecisionTreeClassificationModel
from pyspark.sql.types import StructType, StructField, FloatType, StringType
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
import pandas as pd
import matplotlib.pyplot as plt

# Khởi tạo SparkSession
@st.cache_resource
def get_spark_session():
    return SparkSession.builder \
        .appName("ChurnPredictionApp") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

spark = get_spark_session()

# Đường dẫn mô hình
MODEL_PATH = "/content/drive/MyDrive/BigData_Project/models/decision_tree_model"

# Tải mô hình
@st.cache_resource
def load_model(model_path):
    return DecisionTreeClassificationModel.load(model_path)

model = load_model(MODEL_PATH)

# Tạo giao diện
st.title("Customer Churn Prediction")
st.write("Application to predict customer churn.")

# Nhập thông tin khách hàng
st.header("1. Enter customer information")
with st.form("customer_form"):
    account_length = st.number_input("Account length", min_value=0, value=100)
    international_plan = st.selectbox("International plan", ["Yes", "No"])
    voice_mail_plan = st.selectbox("Voice mail plan", ["Yes", "No"])
    number_vmail_messages = st.number_input("Number of voicemail messages", min_value=0, value=0)
    total_day_minutes = st.number_input("Total day minutes", min_value=0.0, value=150.0)
    total_day_calls = st.number_input("Total day calls", min_value=0, value=50)
    total_eve_minutes = st.number_input("Total evening minutes", min_value=0.0, value=120.0)
    total_eve_calls = st.number_input("Total evening calls", min_value=0, value=40)
    total_night_minutes = st.number_input("Total night minutes", min_value=0.0, value=100.0)
    total_night_calls = st.number_input("Total night calls", min_value=0, value=30)
    total_intl_minutes = st.number_input("Total international minutes", min_value=0.0, value=10.0)
    total_intl_calls = st.number_input("Total international calls", min_value=0, value=3)
    customer_service_calls = st.number_input("Customer service calls", min_value=0, value=1)
    submitted = st.form_submit_button("Predict")

# Dự đoán từ thông tin khách hàng
if submitted:
    # Kiểm tra tất cả các giá trị đầu vào
    inputs = [
        account_length, number_vmail_messages, total_day_minutes, total_day_calls,
        total_eve_minutes, total_eve_calls, total_night_minutes, total_night_calls,
        total_intl_minutes, total_intl_calls, customer_service_calls
    ]

    if all(val == 0 for val in inputs):
        st.error("Requires customer data entry. Please enter at least one value other than 0.")
    else:
        # Chuyển đổi dữ liệu người dùng thành DataFrame của Spark
        input_data = spark.createDataFrame([{
            "Account_length": float(account_length),
            "International_plan": international_plan,
            "Voice_mail_plan": voice_mail_plan,
            "Number_vmail_messages": float(number_vmail_messages),
            "Total_day_minutes": float(total_day_minutes),
            "Total_day_calls": float(total_day_calls),
            "Total_eve_minutes": float(total_eve_minutes),
            "Total_eve_calls": float(total_eve_calls),
            "Total_night_minutes": float(total_night_minutes),
            "Total_night_calls": float(total_night_calls),
            "Total_intl_minutes": float(total_intl_minutes),
            "Total_intl_calls": float(total_intl_calls),
            "Customer_service_calls": float(customer_service_calls),
        }])

        # Xử lý dữ liệu
        indexer_international_plan = StringIndexer(inputCol="International_plan", outputCol="International_plan_index")
        indexer_voice_mail_plan = StringIndexer(inputCol="Voice_mail_plan", outputCol="Voice_mail_plan_index")
        assembler = VectorAssembler(
            inputCols=[
                "Account_length", "International_plan_index", "Voice_mail_plan_index",
                "Number_vmail_messages", "Total_day_minutes", "Total_day_calls", "Total_eve_minutes",
                "Total_eve_calls", "Total_night_minutes", "Total_night_calls", "Total_intl_minutes",
                "Total_intl_calls", "Customer_service_calls"
            ],
            outputCol="features"
        )
        pipeline = Pipeline(stages=[indexer_international_plan, indexer_voice_mail_plan, assembler])
        processed_data = pipeline.fit(input_data).transform(input_data)
        processed_data = processed_data.withColumnRenamed("features", "indexedFeatures")

        # Dự đoán với mô hình
        predictions = model.transform(processed_data)
        churn_prediction = predictions.select("prediction").collect()[0]["prediction"]

# Display prediction results
    st.subheader("Predicted results")
    if churn_prediction == 1.0:
        st.error("Customer is likely to churn.")
        st.write("### Suggestions:")
        suggestions = []
        if account_length > 180:
            suggestions.append("Long-term customer, send a loyalty program.")
        if account_length < 50:
            suggestions.append("New customer, improve their service experience.")
        if international_plan == "No" and total_intl_minutes > 15:
            suggestions.append("Propose an international plan to reduce costs.")
        if voice_mail_plan == "No" and number_vmail_messages > 10:
            suggestions.append("Recommend signing up for voicemail service.")
        if total_day_minutes > 250:
            suggestions.append("Propose an unlimited day plan.")
        if total_eve_minutes > 300:
            suggestions.append("Propose an evening call discount plan.")
        if total_night_minutes > 350:
            suggestions.append("Consider offering a night plan discount.")
        if total_intl_calls > 10:
            suggestions.append("Introduce an international call discount plan.")
        if customer_service_calls > 5:
            suggestions.append("Check the customer's issues and provide timely support.")
    else:
        st.success("Customer is not likely to churn.")
        st.write("### Suggestions:")
        suggestions = []
        if account_length > 180:
            suggestions.append("These are loyal customers, come up with gratitude programs.")
        if account_length < 50:
            suggestions.append("This is a new customer, please offer special incentives and policies to increase service experience and retain customers.")
        if 50 <= total_day_minutes <= 200 and 100 <= total_eve_minutes <= 250 and 100 <= total_night_minutes <= 250:
            suggestions.append("These are customers who regularly use the company's services. Make promotional announcements like the next recharge discount")
        if total_intl_minutes > 10 and international_plan == "No":
            suggestions.append("Propose signing up for an international plan to reduce costs.")
        if customer_service_calls == 0:
            suggestions.append("Participate in our experience survey to receive offers.")

        for i, suggestion in enumerate(suggestions, 1):
            st.write(f"{i}. {suggestion}")

