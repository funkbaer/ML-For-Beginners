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
# model = 0

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/train", methods=["GET"])
def train():
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

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    model = pickle.load(open("../ufo-model.pkl", "rb"))
    prediction = model.predict(final_features)

    output = prediction[0]

    countries = ["Australia", "Canada", "Germany", "UK", "US"]

    return render_template(
        "index.html", prediction_text="Likely country: {}".format(countries[output])
    )




if __name__ == "__main__":
    app.run(debug=True)