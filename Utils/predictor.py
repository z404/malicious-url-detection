import pickle
import pandas as pd
from . import retrainer

def predict(url):
    df = pd.DataFrame({"Unnamed: 0":0, "url": url}, index=[0])
    df = retrainer.preprocess_dataset(df)
    X = df[['length_of_hostname',
        'length_of_path', 'length_of_fd', 'length_of_td', '-cnt', '@cnt', '?cnt',
        '%cnt', '.cnt', '=cnt', 'httpcnt', 'digitcnt', #'httpscnt', 'wwwcnt',
        'lettercnt', 'dircnt', 'is_ip']]
    # Load the model
    model = pickle.load(open('model.pkl', 'rb'))
    # Make prediction
    prediction = model.predict(X)
    # Return prediction
    return prediction