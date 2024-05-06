from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import pandas as pd

app = FastAPI()

#import model
model = joblib.load('./model.pkl')

class Electric_moter_input_parms(BaseModel):
    # profile_id: float
    u_q: float
    coolant: float
    u_d: float
    motor_speed: float
    i_d: float
    i_q: float
    ambient: float
    
@app.get('/')
async def root():
    return{"status":"Online"}

@app.post('/motor_temp/')
async def motor_temp(parms:Electric_moter_input_parms):
    # print("Parameters : " + str(parms.step))
    re = model.predict([[parms.u_q,parms.coolant,parms.u_d,parms.motor_speed,parms.i_d,parms.i_q,parms.ambient]])
    # print(re)         i_d, i_q, u_q, u_d, ambient, coolant, motor_speed
    return {"PM ": str(re[0])}
