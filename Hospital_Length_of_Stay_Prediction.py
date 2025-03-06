# import libraries 

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 


from sklearn.model_selection import train_test_split,GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OrdinalEncoder , LabelEncoder
from sklearn.metrics import confusion_matrix,mean_squared_error,classification_report,accuracy_score
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier

 
from scipy.stats import randint


import warnings 
warnings.filterwarnings("ignore")



# load data 

df = pd.read_csv("C:\\Users\\Asus\\Desktop\\CALISMALAR\\python\\datasets\\CSV\\Hospital_Inpatient_Discharges__SPARCS_De-Identified___2021_20231012.csv")
df_ = df.head(50)

def info(df1):
    df1.info()
    
    describe = df1.describe()
    print(describe)

info(df)

los = df["Length of Stay"]
df["Length of Stay"] = df["Length of Stay"].replace("120 +",120)
df["Length of Stay"] = pd.to_numeric(df["Length of Stay"])

los = df["Length of Stay"] 

# print(df.isnull().sum())

df = df[df["Patient Disposition"] != "Expired"] # iptal edilmiş olanlar veri setinden çıkarıldı

def uniqeu_value(df1):
    for col in df1.columns :
        unique_values = len(df1[col].unique())
        print(f"number of unique values in {col} :{unique_values}")
        

# uniqeu_value(df)

# EDA
"""
    Length of Stay  =  Age - Type of Admission - Payment Typolog 
"""
def visualization(df):
    sns.boxplot(data = df , x= "Length of Stay", y = "Age Group")
    plt.title("Length of Stay vs Age ")
    plt.xlabel("Length of Stay")
    plt.ylabel("Age")
    plt.show()
  
    sns.boxplot(data = df , x= "Length of Stay", y = "Payment Typology 1")
    plt.xlabel("Length of Stay")
    plt.title("Length of Stay vs Payment Typolog")
    plt.ylabel("Payment Typolog")
    plt.show()
    
    sns.countplot(data = df[df["Payment Typology 1"] == "Medicare"] , x = "Age Group")
    plt.title("Medicare Count")
    plt.show()
    
    
    
    sns.boxplot(data = df , x= "Length of Stay", y = "Type of Admission")
    plt.title("Length of Stay vs Type of Admission ")
    plt.xlabel("Length of Stay")
    plt.ylabel("Type of Admission")
    plt.show()
    
    
# visualization(df)




# feature encoding - selection (label encoding)

df = df.drop(["Hospital Service Area","Hospital County","Operating Certificate Number","Facility Name",
              "Zip Code - 3 digits","Patient Disposition","Discharge Year",
              "CCSR Diagnosis Description", "CCSR Procedure Description","APR DRG Description","APR MDC Description",
              "Payment Typology 2","Payment Typology 3","Birth Weight"
              ],axis = 1 )

def outlier_igr(df1):

    for col in df1.select_dtypes(include= ["int64","float64"]).columns :     
        Q1 = df1[col].quantile(0.25)
        Q3 = df1[col].quantile(0.75)
        
        iqr = Q3 - Q1 
        
        lower_bound = Q1 - (iqr * 1.5)
        upper_bound = Q3 + (iqr * 1.5)
        
        new_df = df[(df[col] > lower_bound ) & (df[col] < upper_bound)]
        return new_df 
    
df = outlier_igr(df)
 

# outlier islemini bundan önce yaz 
def missing_value_new(df1): 
   
    df1["Permanent Facility Id"]= df1["Permanent Facility Id"].fillna(df1["Permanent Facility Id"].median() )
    df1["CCSR Diagnosis Code"] = df1["CCSR Diagnosis Code"].fillna(df1["CCSR Diagnosis Code"].mode()[0])
    df1["CCSR Procedure Code"] = df1["CCSR Procedure Code"].fillna(df1["CCSR Procedure Code"].mode()[0])
    df1["APR Severity of Illness Description"] = df1["APR Severity of Illness Description"].fillna(df1["APR Severity of Illness Description"].mode()[0])
    df1["APR Risk of Mortality"]= df1["APR Risk of Mortality"].fillna(df1["APR Risk of Mortality"].mode()[0])
    return df1 
    
df= missing_value_new(df)    



Encoder_Label = LabelEncoder()

df["Age Group"] = Encoder_Label.fit_transform(np.asarray(df["Age Group"]).reshape(-1,1).ravel())

df["Gender"] = Encoder_Label.fit_transform(np.asarray(df["Gender"]).reshape(-1,1).ravel())

df["Race"] = Encoder_Label.fit_transform(np.asarray(df["Race"]).reshape(-1,1).ravel())

df["Ethnicity"] = Encoder_Label.fit_transform(np.asarray(df["Ethnicity"]).reshape(-1,1).ravel())

df["Type of Admission"] = Encoder_Label.fit_transform(np.asarray(df["Type of Admission"]).reshape(-1,1).ravel())

df["CCSR Diagnosis Code"] = Encoder_Label.fit_transform(np.asarray(df["CCSR Diagnosis Code"]).reshape(-1,1).ravel())

df["Ethnicity"] = Encoder_Label.fit_transform(np.asarray(df["Ethnicity"]).reshape(-1,1).ravel())

df["CCSR Procedure Code"] = Encoder_Label.fit_transform(np.asarray(df["CCSR Procedure Code"]).reshape(-1,1).ravel())

df["APR Medical Surgical Description"] = Encoder_Label.fit_transform(np.asarray(df["APR Medical Surgical Description"]).reshape(-1,1).ravel())

df["Payment Typology 1"] = Encoder_Label.fit_transform(np.asarray(df["Payment Typology 1"]).reshape(-1,1).ravel())

df["Emergency Department Indicator"] = Encoder_Label.fit_transform(np.asarray(df["Emergency Department Indicator"]).reshape(-1,1).ravel())


df["Total Charges"] = pd.to_numeric(df["Total Charges"].str.replace(",",""),errors="coerce" ) # string olarak verilmiş numeric değerlerin donusumu 

df["Total Costs"] = pd.to_numeric(df["Total Costs"].str.replace(",","") ,errors="coerce")


Encoder_Ordinal = OrdinalEncoder()

df["APR Severity of Illness Description"] = Encoder_Ordinal.fit_transform(np.asarray(df["APR Severity of Illness Description"]).reshape(-1,1)) 

df["APR Risk of Mortality"] = Encoder_Ordinal.fit_transform(np.asarray(df["APR Risk of Mortality"]).reshape(-1,1) )



# missing value control (eksik verilerin silinmesi ile )

def missing_value_drop(df):
    print(df.isnull().sum())
    df = df.drop(["CCSR Procedure Code"],axis = 1 )
    df = df.dropna(subset =["APR Risk of Mortality","Permanent Facility Id","APR Severity of Illness Description"])    
    
    return df 

df = missing_value_drop(df)



def box_plot(df1):
    plt.boxplot(df1)
    plt.title("boxplot")
    plt.show()
    
    corr = df1.corr()
    plt.figure(figsize = (10,10))
    sns.heatmap(data = corr ,annot = True, fmt = ".2f")
    plt.title("Corelation matrix ")
    plt.show()

# box_plot(df)


# train test split 

x = df.drop(["Length of Stay"],axis = 1 )
y = df["Length of Stay"]
    

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size = 0.2)


def Decision_tree(x_train, x_test, y_train, y_test):
    # regression : train and test 
    dtree = DecisionTreeRegressor(max_depth = 10 )

    #default parametlerle model
    dtree.fit(x_train, y_train)
    train_pred = dtree.predict(x_train)
    test_pred = dtree.predict(x_test)

    # RMSE hesaplama
    print("Decision_tree_train RMSE:", np.sqrt(mean_squared_error(y_train, train_pred)))
    print("Decision_tree_test  RMSE:", np.sqrt(mean_squared_error(y_test, test_pred)))

    # Hiperparametre grid'i
    param_grid_reg = {
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    # GridSearchCV ile hiperparametre optimizasyonu
    grid_search_reg = GridSearchCV(estimator=dtree, param_grid=param_grid_reg,
                                   cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search_reg.fit(x_train, y_train)

    # En iyi parametreler
    print("Best Parameters for Decision Tree Regressor:", grid_search_reg.best_params_)

    # En iyi model
    best_dtree_reg = grid_search_reg.best_estimator_

    # Tahminler
    train_pred_reg = best_dtree_reg.predict(x_train)
    test_pred_reg = best_dtree_reg.predict(x_test)


    # RMSE hesaplama
    print("Decision Tree Regressor Train RMSE:", np.sqrt(mean_squared_error(y_train, train_pred_reg)))
    print("Decision Tree Regressor Test RMSE:", np.sqrt(mean_squared_error(y_test, test_pred_reg)))
    
    
    """
    
    RMSE: 0.06333433516415934
    RMSE: 3.5187629025188865
    
    
    after max_depth = 10 
    RMSE: 3.3151806237468433
    RMSE: 3.4736216232797723
    
    """

    
    # veriyi kategorik hale getirme solve classification problem train test 
    

    
    bins = [0, 5, 10, 20, 30, 50, 120]
    labels = [ 5, 10, 20, 30, 50, 120]
    
    
    
    df["los_bin"] = pd.cut(x = df["Length of Stay"],bins = bins )
    df["los_labels"] = pd.cut(x = df["Length of Stay"], bins = bins, labels = labels)
    
    df_ = df.head(50)
    
    df["los_bin"] = df["los_bin"].apply(lambda x : str(x).replace(",","-"))
    df["los_bin"] = df["los_bin"].apply(lambda x : str(x).replace("120","120+"))
    
    f,ax = plt.subplots()
    sns.countplot(x = "los_bin",data = df)
    
    new_x = df.drop(["Length of Stay","los_bin","los_labels"],axis = 1 )
    new_y = df["los_labels"]
    
    
    x_train, x_test, y_train, y_test = train_test_split(new_x, new_y, random_state=42, test_size = 0.2)

    dtree_clf_default = DecisionTreeClassifier(random_state=42)
    dtree_clf_default.fit(x_train, y_train)

    # Varsayılan model için tahminler
    train_pred_default_clf = dtree_clf_default.predict(x_train)
    test_pred_default_clf = dtree_clf_default.predict(x_test)

    # Varsayılan model için performans metrikleri
    print("Default Decision Tree Classifier Train Accuracy:", accuracy_score(y_train, train_pred_default_clf))
    print("Default Decision Tree Classifier Test Accuracy:", accuracy_score(y_test, test_pred_default_clf))
    print("Default Decision Tree Classifier Confusion Matrix:\n", confusion_matrix(y_test, test_pred_default_clf))
    print("Default Decision Tree Classifier Classification Report:\n",
          classification_report(y_test, test_pred_default_clf))


    #GridSearchCV ile hiperparametre optimizasyonu
    dtree_clf = DecisionTreeClassifier(random_state=42)

    param_grid_clf = {
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    grid_search_clf = GridSearchCV(estimator=dtree_clf, param_grid=param_grid_clf,
                                   cv=5, scoring='accuracy', n_jobs=-1)
    grid_search_clf.fit(x_train, y_train)

    # En iyi parametreler
    print("Best Parameters for Decision Tree Classifier:", grid_search_clf.best_params_)

    # En iyi model
    best_dtree_clf = grid_search_clf.best_estimator_

    # En iyi model için tahminler
    train_pred_best_clf = best_dtree_clf.predict(x_train)
    test_pred_best_clf = best_dtree_clf.predict(x_test)

    # En iyi model için performans metrikleri
    print("Optimized Decision Tree Classifier Train Accuracy:", accuracy_score(y_train, train_pred_best_clf))
    print("Optimized Decision Tree Classifier Test Accuracy:", accuracy_score(y_test, test_pred_best_clf))
    print("Optimized Decision Tree Classifier Confusion Matrix:\n", confusion_matrix(y_test, test_pred_best_clf))
    print("Optimized Decision Tree Classifier Classification Report:\n",
          classification_report(y_test, test_pred_best_clf))


    
Decision_tree(x_train, x_test, y_train, y_test)



def Random_Forest(x_train, x_test, y_train, y_test):
    print("RANDOM FOREST \n")
    rf = RandomForestClassifier()


    # model parametre template 
    
    param_list = {
        'n_estimators': randint(10, 50),  # 10 ile 50 arasında rastgele  n_estimators
        'max_depth': [None, 10, 20, 30, 40],  # Farklı derinlik seçenekleri
        'min_samples_split': randint(2, 10),  # 2 ile 10 arasında rastgele seçilecek min_samples_split
        'min_samples_leaf': randint(1, 10),   # 1 ile 10 arasında rastgele seçilecek min_samples_leaf
        'max_features': ['auto', 'sqrt', 'log2'],  # Max feature seçenekleri
        }
    
    random_search = RandomizedSearchCV(estimator = rf, param_distributions = param_list , cv = 2, n_iter=5,random_state = 42  )
    random_search.fit(x_train, y_train)
    
    print(random_search.best_params_)
    
    best_params_rf = random_search.best_estimator_
    
    best_params_rf.fit(x_train, y_train)
    train_pred = best_params_rf.predict(x_train)
    test_pred = best_params_rf.predict(x_test)

     
    print("Random_forest_train RMSE:",np.sqrt(mean_squared_error(y_train, train_pred)))
    print("Random_forest_test RMSE:",np.sqrt(mean_squared_error(y_test, test_pred)))
    
    
    cm = confusion_matrix(y_test, test_pred)
    
    print("confusion matrix")
    
    print(cm,"\n \n")
    
    print("train accuracy : ",accuracy_score(y_train,train_pred))
    
    print("test accuracy : ",accuracy_score(y_test,test_pred))
    
    print("Classification report :", classification_report(y_test, test_pred))
    

Random_Forest(x_train, x_test, y_train, y_test)
















