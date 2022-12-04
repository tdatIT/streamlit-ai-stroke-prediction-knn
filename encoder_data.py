numerical_cols=['age', 'avg_glucose_level', 'bmi']
import pandas as pd
binary_encoder_cols=['hypertension', 'heart_disease', 'ever_married']
from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()

df=pd.read_csv("./brain_stroke.csv")
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
dict_cs={}
for col in numerical_cols:
    sc=sc.fit(df[[col]])
    dict_cs[col]=sc
for col in binary_encoder_cols:
    lb=lb.fit(df[[col]])

    dict_cs[col]=lb

def new_data_num(data):
    for col in numerical_cols:
        data[col]=dict_cs[col].transform(data[[col]])
    for col in binary_encoder_cols:
        data[col]=dict_cs[col].transform(data[col])
    return data
    
