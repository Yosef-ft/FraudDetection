import joblib
from .utils import Utils

model = joblib.load('../mlartifacts/891150804028438362/6394144cb4b54e858375ecacec0d1094/artifacts/Fraud_RF_path/model.pkl')

def predict(data):
    utils = Utils()
    data = utils.preprocess(data)
 
    prediction = model.predict(data)
    return prediction
