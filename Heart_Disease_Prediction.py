# import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

import statsmodels.api as sm
from sklearn.impute import KNNImputer
from scipy import stats

import warnings

warnings.filterwarnings("ignore")
import os

# load data and EDA

# BASE_DIR dinamik olarak ayarlanır
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# CSV dosyasının yolunu dinamik olarak belirleriz
csv_path = os.path.join(BASE_DIR, "datasets", "CSV",
                        "Hospital_Inpatient_Discharges__SPARCS_De-Identified___2021_20231012.csv")

df = pd.read_csv(csv_path)

df = df.drop(columns="id")


def info(df):
    df.info()  # boolean türler object yani string olarak görünüyor

    describe = df.describe()

    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()

    sns.pairplot(df, vars=numeric_features, hue="num")
    plt.show()

    sns.countplot(x="num", data=df)
    plt.show()


# handling missing value


def missing_value(df1):
    print("missing value function :", df1.isnull().sum())  # columnslardaki missing value sayıları

    df1["fbs"] = df1["fbs"].fillna(
        df["fbs"].mode()[0])  # fbs True ya da false turunden bir feature bu yüzden mod alında en çok olan değer yazılır
    df1["restecg"] = df1["restecg"].fillna(df["restecg"].mode()[0])
    df1["exang"] = df1["exang"].fillna(df["exang"].mode()[0])
    df1["slope"] = df1["slope"].fillna(df["slope"].mode()[0])
    df1["thal"] = df1["thal"].fillna(df["thal"].mode()[0])
    df1["thalch"] = df1["thalch"].fillna(df["thalch"].mode()[0])

    return df1


df = missing_value(df)


def detect_outliers_iqr(df1):
    outlier_indices = []
    outlier_df = pd.DataFrame()

    for col in df1.select_dtypes(include=["float64", "int64"]).columns:
        Q1 = df1[col].quantile(0.25)  # first quartile
        Q3 = df1[col].quantile(0.75)  # third quartile
        igr = Q3 - Q1

        lower_bound1 = Q1 - 1.5 * igr
        upper_bound1 = Q3 + 1.5 * igr

        outlier_in_col = df1[(df1[col] < lower_bound1) | (df1[col] > upper_bound1)]

        outlier_indices.extend(outlier_in_col.index)
        outlier_df = pd.concat([outlier_df, outlier_in_col], axis=0)
    # remove duplicate indices
    outlier_indices = list(set(outlier_indices))

    # remove duplicate rows in the outlier Dataframe
    outlier_df = outlier_df.drop_duplicates()
    return outlier_df


# df = detect_outliers_iqr(df)   bu outlier hesaplama ile accuracy degeri dustu


# detect numeric outlier values
def detect_outlier_zscore(df1):
    z_scores_dict = {}
    for col in df1.columns:
        mean = np.mean(df1[col])
        std = np.std(df1[col])

        z_score = [(x - mean) / std for x in df1[col]]

        z_scores_dict[col] = z_score

    z_score_df = pd.DataFrame(z_scores_dict, columns=df1.columns, index=df1.index)
    return z_score_df


categorical_features = ["sex", "dataset", "cp", "exang", "restecg", "slope", "thal", "fbs"]
numerical_features = ["trestbps", "chol", "thalch", "oldpeak", "age"]


def KNN_imputer_missing_values(df1):
    print("kontrol1", df1.isnull().sum())
    knnimputer = KNNImputer(n_neighbors=5)
    df_imputed = knnimputer.fit_transform(df1)
    df_imputed1 = pd.DataFrame(data=df_imputed, columns=df1.columns, index=df1.index)
    print("kontrol2", df_imputed1.isnull().sum())
    return df_imputed1


x = df.drop(["num"], axis=1)
y = df["num"]

x_categorical = x[categorical_features]

x_numerical_imputed = KNN_imputer_missing_values(x[numerical_features])

x_numerical_imputed = pd.DataFrame(data=x_numerical_imputed, columns=numerical_features)

outlier_zscore = detect_outlier_zscore(x_numerical_imputed)
outliers = (abs(outlier_zscore > 3))

x_numerical_imputed_no_outliers = x_numerical_imputed[~outliers.any(axis=1)]
x_categorical = x_categorical[~outliers.any(axis=1)]

y = y[~outliers.any(axis=1)]

x_result = pd.concat([x_categorical.reset_index(drop=True), x_numerical_imputed_no_outliers.reset_index(drop=True)],
                     axis=1)

# train test split - standardizasyon - kategorik kodlama

x_train, x_test, y_train, y_test = train_test_split(x_result, y, test_size=0.25, random_state=42)

x_train_num = x_train[numerical_features]
x_test_num = x_test[numerical_features]

scaler = StandardScaler()
x_train_num_scaled = scaler.fit_transform(x_train_num)
x_test_num_scaled = scaler.transform(x_test_num)

encoded = OneHotEncoder(drop="first", sparse_output=False)

x_train_cat = x_train[categorical_features]
x_test_cat = x_test[categorical_features]

x_train_cat_encoded = encoded.fit_transform(x_train_cat)
x_test_cat_encoded = encoded.transform(x_test_cat)

x_train_transformed = np.hstack((x_train_cat_encoded, x_train_num_scaled))
x_test_transformed = np.hstack((x_test_cat_encoded, x_test_num_scaled))

# modelling = RF, KNN, Voting Classifier train test
rf = RandomForestClassifier(n_estimators=10, random_state=42)
knn = KNeighborsClassifier(n_neighbors=10)

voting_clf = VotingClassifier(estimators=[
    ("rf", rf),
    ("knn", knn)], voting="soft")

# Model karsılastırma


# model egitimi
voting_clf.fit(x_train_transformed, y_train)

# test verisi ile tahmin
y_pred = voting_clf.predict(x_test_transformed)
accuracy = accuracy_score(y_test, y_pred)
print("accuracy: ", accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("confusion matrix ")
print(cm)

print("classificatin report: ", classification_report(y_test, y_pred))

# CM with seaborn

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion matrix")
plt.xlabel("Predicted")
plt.ylabel("True label")
plt.show()

# summary and p - value

model = sm.OLS(y_train, x_train_transformed).fit()
summary = model.summary()

print(summary)