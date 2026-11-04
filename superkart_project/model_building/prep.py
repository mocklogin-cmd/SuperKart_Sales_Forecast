import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("superkart_project/data/SuperKart.csv")
print("Raw shape:", df.shape)

# 1. fix the wrong label - 'reg' is the same thing as 'Regular'
df["Product_Sugar_Content"] = df["Product_Sugar_Content"].replace({"reg": "Regular"})

# 2. remove duplicate rows (there are none in this file, but the pipeline should still be safe)
df = df.drop_duplicates()

# 3. Product_Id starts with FD / DR / NC, so we keep that as a simple category
#    FD = Food, DR = Drinks, NC = Non-Consumable
df["Product_Category"] = df["Product_Id"].str[:2].map(
    {"FD": "Food", "DR": "Drinks", "NC": "Non-Consumable"}
)

# 4. drop columns we do not need
#    Product_Id is unique for every row so the model cannot learn from it
#    Store_Id is already described by size / city type / store type / year
df = df.drop(columns=["Product_Id", "Store_Id"])

# NOTE: categorical columns are left as text on purpose. They get one-hot-encoded
# inside the model pipeline so training and the Streamlit app use the same steps.

target = "Product_Store_Sales_Total"
X = df.drop(columns=[target])
y = df[target]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Cleaned shape:", df.shape)
print("Train rows:", Xtrain.shape[0], "| Test rows:", Xtest.shape[0])
print("Sugar content values:", sorted(df["Product_Sugar_Content"].unique()))
print("Columns used for modelling:", list(X.columns))
print("Train / test splits saved.")
