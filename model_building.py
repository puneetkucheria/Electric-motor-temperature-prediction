import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors
from sklearn.metrics import mean_squared_error ,  mean_absolute_percentage_error, mean_absolute_error
from math import sqrt
from sklearn.metrics import r2_score

def train_test_split_and_scale(df):
    y = df['pm']
    x = df.drop('pm',axis=1)
    features = list(x.columns)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state = 0)
    # scaler = MinMaxScaler()
    # x_train = scaler.fit_transform(x_train)
    # x_test = scaler.transform(x_test)
    return x_train, x_test, y_train, y_test,features #, scaler

def fit_and_evaluate_model(x_train, x_test, y_train, y_test,xgb):
    rmse_val = [] #to store rmse values for different k
    # k_val = []
    r_squared_score = []
    xgb.fit(x_train, y_train)
    xgb_predict = xgb.predict(x_test)
    error = sqrt(mean_squared_error(y_test,xgb_predict)) #calculate rmse
    rmse_val.append(error) #store rmse values
    r2_val = r2_score(y_test, xgb_predict)
    r_squared_score.append(r2_val)
    # k_val.append(K)
    # print('RMSE value for k= ' , K , 'is:', error)
    print("R-squared score: ", r2_val)
    return xgb