

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("heart.csv")

""" Check for null values And processedv

"""

data.info()

missing_values = data.isnull().sum()
print("Missing values per column:\n", missing_values)
for column in data.columns:
    if data[column].isnull().sum() > 0:
        if data[column].dtype == 'float64' or data[column].dtype == 'int64':
            data[column].fillna(data[column].median(), inplace=True)
        else:            data[column].fillna(data[column].mode()[0], inplace=True)

"""Draw box plots before handling outliers"""

plt.figure(figsize=(15, 8))
for i, column in enumerate(data.select_dtypes(include=[np.number]).columns, 1):
    plt.subplot(2, len(data.select_dtypes(include=[np.number]).columns)//2 + 1, i)
    sns.boxplot(x=data[column], color='skyblue')
    plt.title(f'Before Cleaning: {column}')
plt.tight_layout()
plt.show()

"""Handling outliers"""

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
for column in data.select_dtypes(include=[np.number]).columns:
    data = remove_outliers_iqr(data, column)
data.describe()

cleaned_data = data.copy()


for column in cleaned_data.select_dtypes(include=[np.number]).columns:
    cleaned_data = remove_outliers_iqr(cleaned_data, column)

plt.figure(figsize=(15, 8))
for i, column in enumerate(cleaned_data.select_dtypes(include=[np.number]).columns, 1):
    plt.subplot(2, len(cleaned_data.select_dtypes(include=[np.number]).columns)//2 + 1, i)
    sns.boxplot(x=cleaned_data[column], color='lightgreen')
    plt.title(f'After Cleaning: {column}')
plt.tight_layout()
plt.show()

"""#1. What is the age distribution among patients?"""

plt.figure(figsize=(10, 6))
sns.histplot(data['age'], kde=True, color="skyblue", bins=20)
plt.title("Distribution of Age")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

""" What is the ratio of males to females in the data set?"""

plt.figure(figsize=(6, 6))
sns.countplot(x='sex', data=data, palette="viridis")
plt.title("Gender Distribution")
plt.xlabel("Gender (0 = Female, 1 = Male)")
plt.ylabel("Count")
plt.show()

"""How are heart patients distributed by type of chest pain (cp)?"""

plt.figure(figsize=(8, 6))
sns.countplot(x='cp', hue='output', data=data, palette="Set2")
plt.title("Chest Pain Type Distribution by Heart Disease")
plt.xlabel("Chest Pain Type (cp)")
plt.ylabel("Count")
plt.legend(title="Heart Disease (0 = No, 1 = Yes)")
plt.show()

"""What is the relationship between age and blood pressure (trtbps)?"""

plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='trtbps', hue='output', data=data, palette="coolwarm")
plt.title("Relation between Age and Resting Blood Pressure")
plt.xlabel("Age")
plt.ylabel("Resting Blood Pressure (trtbps)")
plt.legend(title="Heart Disease")
plt.show()

""" What is the distribution of cholesterol level (chol)?"""

plt.figure(figsize=(10, 6))
sns.histplot(data['chol'], kde=True, color="lightcoral", bins=20)
plt.title("Distribution of Cholesterol Levels")
plt.xlabel("Cholesterol (mg/dl)")
plt.ylabel("Frequency")
plt.show()

""" What is the relationship between cholesterol and age?"""

plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='chol', hue='output', data=data, palette="viridis")
plt.title("Relation between Age and Cholesterol")
plt.xlabel("Age")
plt.ylabel("Cholesterol (mg/dl)")
plt.legend(title="Heart Disease")
plt.show()

""" What is the level of maximum heart rate (thalachh) in patients with heart disease versus others?"""

plt.figure(figsize=(10, 6))
sns.boxplot(x='output', y='thalachh', data=data, palette="coolwarm")
plt.title("Maximum Heart Rate (thalachh) by Heart Disease")
plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
plt.ylabel("Maximum Heart Rate (thalachh)")
plt.show()

""" How is fasting blood glucose (FBS) distributed among patients?"""

plt.figure(figsize=(6, 6))
sns.countplot(x='fbs', data=data, palette="Set1")
plt.title("Fasting Blood Sugar Distribution (fbs)")
plt.xlabel("Fasting Blood Sugar > 120 mg/dl (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

""" How do the results of an electrocardiogram (restecg) affect the presence of heart disease?"""

plt.figure(figsize=(8, 6))
sns.countplot(x='restecg', hue='output', data=data, palette="Set3")
plt.title("Resting ECG Results by Heart Disease")
plt.xlabel("Resting ECG (restecg)")
plt.ylabel("Count")
plt.legend(title="Heart Disease (0 = No, 1 = Yes)")
plt.show()

""" What is the relationship between ST depression during exercise (oldpeak) and the presence of heart disease?"""

plt.figure(figsize=(10, 6))
sns.violinplot(x='output', y='oldpeak', data=data, palette="muted")
plt.title("ST Depression (oldpeak) by Heart Disease")
plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
plt.ylabel("ST Depression (oldpeak)")
plt.show()