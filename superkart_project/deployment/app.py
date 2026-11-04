import os
import streamlit as st
import pandas as pd
import joblib

# the model is committed to the repo by the GitHub Actions pipeline
model_path = os.path.join(os.path.dirname(__file__), "best_superkart_sales_model_v1.joblib")
model = joblib.load(model_path)

# the 4 SuperKart stores and their details (taken from the dataset)
STORES = {
    "OUT001": {"Store_Establishment_Year": 1987, "Store_Size": "High",
               "Store_Location_City_Type": "Tier 2", "Store_Type": "Supermarket Type1"},
    "OUT002": {"Store_Establishment_Year": 1998, "Store_Size": "Small",
               "Store_Location_City_Type": "Tier 3", "Store_Type": "Food Mart"},
    "OUT003": {"Store_Establishment_Year": 1999, "Store_Size": "Medium",
               "Store_Location_City_Type": "Tier 1", "Store_Type": "Departmental Store"},
    "OUT004": {"Store_Establishment_Year": 2009, "Store_Size": "Medium",
               "Store_Location_City_Type": "Tier 2", "Store_Type": "Supermarket Type2"},
}

PRODUCT_TYPES = [
    "Baking Goods", "Breads", "Breakfast", "Canned", "Dairy", "Frozen Foods",
    "Fruits and Vegetables", "Hard Drinks", "Health and Hygiene", "Household",
    "Meat", "Others", "Seafood", "Snack Foods", "Soft Drinks", "Starchy Foods",
]

st.title("SuperKart Sales Forecast")
st.write("Enter the product and store details to predict the total sales "
         "of that product in that store.")

# ---- product inputs ----
st.subheader("Product details")
Product_Weight = st.number_input("Product weight", min_value=1.0, max_value=30.0, value=12.5, step=0.1)
Product_Sugar_Content = st.selectbox("Sugar content", ["Low Sugar", "Regular", "No Sugar"])
Product_Allocated_Area = st.number_input(
    "Allocated display area (ratio)", min_value=0.001, max_value=0.400, value=0.060, step=0.001, format="%.3f")
Product_Type = st.selectbox("Product type", PRODUCT_TYPES)
Product_Category = st.selectbox("Product category", ["Food", "Drinks", "Non-Consumable"])
Product_MRP = st.number_input("Product MRP", min_value=10.0, max_value=300.0, value=147.0, step=1.0)

# ---- store inputs ----
st.subheader("Store details")
store_id = st.selectbox("Store", list(STORES.keys()))
store = STORES[store_id]
st.caption(
    f"{store['Store_Type']} | {store['Store_Size']} size | "
    f"{store['Store_Location_City_Type']} | established {store['Store_Establishment_Year']}"
)

# put the inputs into a dataframe (same column names as used in training)
input_data = pd.DataFrame([{
    "Product_Weight": Product_Weight,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_Type": Product_Type,
    "Product_MRP": Product_MRP,
    "Store_Establishment_Year": store["Store_Establishment_Year"],
    "Store_Size": store["Store_Size"],
    "Store_Location_City_Type": store["Store_Location_City_Type"],
    "Store_Type": store["Store_Type"],
    "Product_Category": Product_Category,
}])

st.write("Input sent to the model:")
st.dataframe(input_data)

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Product Store Sales Total: {prediction:,.2f}")
