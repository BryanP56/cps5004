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

