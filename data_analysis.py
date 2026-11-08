import pandas as pd

df = pd.read_csv("wdbc.data", header=None)

df.columns = (
    ["id", "diagnosis"]
    + [f"feature_{i}" for i in range(1, 31)]
)

print(df.head())
print(df.shape)
print(df.isnull().sum())
print(df.info())
print(df.describe())

# STEP 3: Data Preprocessing

# Convert diagnosis labels into numerical values
df["diagnosis"] = df["diagnosis"].map({
    "B": 0,
    "M": 1
})

# Separate input features and target variable
X = df.drop(columns=["id", "diagnosis"])
y = df["diagnosis"]

print("Features:", X.shape)
print("Target:", y.shape)

# STEP 4: Split the dataset into Training and Testing Data

from sklearn.model_selection import train_test_split

# Split dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Display training and testing dataset sizes
print("\nTraining Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)

print("Training Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)

# Save test dataset for Streamlit application
test_data = X_test.copy()
test_data["diagnosis"] = y_test

test_data.to_csv("test_data.csv", index=False)

print("\nTest dataset saved successfully!")