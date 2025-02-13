import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load datasets
data_path = "C:/Programming/Vendor Risk Assessment & Anomaly Detection/"

df_main = pd.read_csv(data_path + "DataCoSupplyChainDataset.csv", encoding='ISO-8859-1')
df_desc = pd.read_csv(data_path + "DescriptionDataCoSupplyChain.csv", encoding='ISO-8859-1')
df_logs = pd.read_csv(data_path + "tokenized_access_logs.csv", encoding='ISO-8859-1')

# Select important columns for analysis
selected_columns = ['Late_delivery_risk', 'Sales', 'Order Status', 'Shipping Mode',
                    'Order Profit Per Order', 'Order Item Total', 'Order Item Quantity', 'Category Name']
df = df_main[selected_columns].copy()

# Encode categorical variables
le = LabelEncoder()
df['Order Status'] = le.fit_transform(df['Order Status'])
df['Shipping Mode'] = le.fit_transform(df['Shipping Mode'])
df['Category Name'] = le.fit_transform(df['Category Name'])

# Handle missing values
df.dropna(inplace=True)

# Normalize numerical features
scaler = StandardScaler()
numeric_cols = ['Sales', 'Order Profit Per Order', 'Order Item Total', 'Order Item Quantity']
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Train-test split
X = df.drop('Late_delivery_risk', axis=1)
y = df['Late_delivery_risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train ML models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(),
    "XGBoost": XGBClassifier(eval_metric='logloss')
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("-" * 50)

# Feature Importance from Random Forest
rf = models["Random Forest"]
feature_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8,5))
sns.barplot(x=feature_importance, y=feature_importance.index)
plt.title("Feature Importance")
plt.show()

# Anomaly Detection using Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
df['Anomaly_Score'] = iso_forest.fit_predict(X)
df['Risk Score'] = -df['Anomaly_Score']  # Higher means more risky

# Rank vendors based on risk
ranked_vendors = df.sort_values(by='Risk Score', ascending=False)
ranked_vendors.to_csv(data_path + "Ranked_Vendor_Risk.csv", index=False)
print("Final vendor risk ranking saved!")