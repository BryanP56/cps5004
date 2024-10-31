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
