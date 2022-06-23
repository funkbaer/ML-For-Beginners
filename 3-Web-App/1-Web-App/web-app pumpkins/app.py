import numpy as np
from flask import Flask, request, render_template
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
import pandas as pd

app = Flask(__name__)

#model = pickle.load(open("../ufo-model.pkl", "rb"))
model = pickle.load(open("../pumpkin_model.pkl", "rb"))
label_encoder_dct = pickle.load(open("../label_encoder_dct.pkl", "rb"))

# model = 0

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/train", methods=["GET"])
def train():
    
    pumpkins = pd.read_csv('../data/US-pumpkins.csv')

    # Reduce dataset to relevant features:
    new_columns = ['Color','Origin','Item Size']
    new_pumpkins = pumpkins.drop([c for c in pumpkins.columns if c not in new_columns], axis=1)
    new_pumpkins.dropna(inplace=True)
    # Create translation dictionary with unique values of each column and
    # corresponding labels:
    trans_dct = {}
    for c in new_pumpkins.columns:
        trans_dct[c] = new_pumpkins[c].unique()
    
    ufos = pd.read_csv('../data/ufos.csv')
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    ufos.dropna(inplace=True)
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])

    Selected_features = ['Seconds','Latitude','Longitude']
    X = ufos[Selected_features]
    y = ufos['Country']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    pickle.dump(model, open("../ufo-model.pkl",'wb'))
    #predictions = model.predict(X_test)
    return render_template(
        "index.html"
    )

@app.route("/predict", methods=["POST"])
def predict():

    #int_features = [int(x) for x in request.form.values()]
    str_features = [ x for x in request.form.values()]

    
    # model = pickle.load(open("../pumpkin_model.pkl", "rb"))
    label_encoder_dct = pickle.load(open("../label_encoder_dct.pkl", "rb"))
    dct_keys = list(label_encoder_dct.keys())
    labeled_features = []
    for i in range(2):
        labeled_features.append(label_encoder_dct[dct_keys[i]].transform([str_features[i]]))
    

        
    df = pd.DataFrame({"Item Size":labeled_features[0], "City Name":labeled_features[1]})
    print(df.info())
    prediction = model.predict(df)
    print(prediction)
    output = prediction[0]

    colors = ["White", "Orange"]


    # countries = ["Australia", "Canada", "Germany", "UK", "US"]

    return render_template(
        "index.html", prediction_text="Likely color: {}".format(colors[output])
    )





if __name__ == "__main__":
    app.run(debug=True)