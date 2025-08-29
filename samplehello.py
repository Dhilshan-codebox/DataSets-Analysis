import streamlit as st
from pymongo import MongoClient
from datetime import datetime

# MongoDB connection
MONGO_URI = "mongodb+srv://Dhilshan:Dhilshan1@cluster001.ekggxmj.mongodb.net/?retryWrites=true&w=majority&appName=Cluster001"
client = MongoClient(MONGO_URI)
db = client['form_db']
collection = db['user_inputs']

st.title("📝 User Feedback Form")

# Form Input
with st.form("input_form"):
    name = st.text_input("Name")
    email = st.text_input("Email")
    age = st.slider("Age", 10, 100, 25)
    feedback = st.text_area("Your Feedback")
    submitted = st.form_submit_button("Submit")

# Save to DB
if submitted:
    record = {
        "name": name,
        "email": email,
        "age": age,
        "feedback": feedback,
        "timestamp": datetime.now()
    }
    collection.insert_one(record)
    st.success("✅ Data saved successfully!")

# Display stored data
if st.checkbox("Show Submitted Data"):
    data = list(collection.find({}, {"_id": 0}))  # Exclude MongoDB's _id
    if data:
        st.write("### 📋 Stored Entries:")
        st.table(data)
    else:
        st.info("No data available.")
