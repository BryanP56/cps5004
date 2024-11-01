import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from scipy.stats import ttest_ind

df = pd.read_csv("Telco-Customer-Churn.csv")

print(df.head())
print(df.info())

print(df.isnull().sum())

df.dropna(inplace=True)

categorical_features = df.select_dtypes(include=['object']).columns
numerical_features = df.select_dtypes(include=[np.number]).columns

df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

df['Total_Charges'] = df['MonthlyCharges'] * df['tenure']

scaler = StandardScaler()
df[['Total_Charges', 'MonthlyCharges']] = scaler.fit_transform(df[['Total_Charges', 'MonthlyCharges']])

print(df.describe())

plt.figure(figsize=(8, 6))
sns.histplot(df['MonthlyCharges'], kde=True)
plt.title("Distribution of Monthly Charges")
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Monthly Charges by Churn")
plt.show()

churned = df[df['Churn'] == 1]['MonthlyCharges']
non_churned = df[df['Churn'] == 0]['MonthlyCharges']

t_stat, p_value = ttest_ind(churned, non_churned)
print(f"T-statistic: {t_stat}, P-value: {p_value}")

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, max_iter=5000, random_state=0)
    kmeans.fit(df[['MonthlyCharges', 'Total_Charges']])
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10,6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=3, max_iter=5000, random_state=0)
df['Cluster'] = kmeans.fit_predict(df[['MonthlyCharges', 'Total_Charges']])

plt.figure(figsize=(10, 6))
sns.scatterplot(x='MonthlyCharges', y='Total_Charges', hue='Cluster', data=df, palette='viridis')
plt.title("Monthly Charges vs. Total Charges Clustering")
plt.show()

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

y_pred = log_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}")

y_pred_prob = log_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label="Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
