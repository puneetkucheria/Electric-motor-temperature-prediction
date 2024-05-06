from data_processing_features import get_data, fields_to_numeric, clean_data
from model_building import train_test_split_and_scale, fit_and_evaluate_model
import joblib

from xgboost import XGBRegressor




em_df = get_data()
em_df = fields_to_numeric(em_df)
em_df = clean_data(em_df)

numerical_features = ['u_q', 'coolant', 'u_d', 'motor_speed', 'i_d', 'i_q', 'ambient', 'pm']
df=em_df[numerical_features]

x_train, x_test, y_train, y_test,features= train_test_split_and_scale(df)
print(features)
parmeters = {'colsample_bytree': 1,
             'learning_rate': 0.3,
             'max_depth': 10,
             'n_estimators': 150,
             'subsample': 1}
xgb =  XGBRegressor(random_state=0, **parmeters)#, device = 'gpu:0')
model = fit_and_evaluate_model(x_train, x_test, y_train, y_test,xgb)

joblib.dump(model, 'model.pkl')