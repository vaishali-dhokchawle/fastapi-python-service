import pandas as pd
import joblib as jb
from sklearn.linear_model import LinearRegression

data=pd.read_csv("bms_data.csv")
X = data[["temperature","humidity",	"co2",	"energy",	"hvac_on","occupancy"]]
y = data["temperature"]
lr_model=LinearRegression()
lr_model.fit(X,y)
jb.dump(lr_model,"bms_lr_model.pkl")
print("Model trained successfully")