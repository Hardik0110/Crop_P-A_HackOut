from flask import Flask,request, render_template
import numpy as np
import pickle
import sklearn
print(sklearn.__version__)
#loading models
dtr = pickle.load(open('dtr.pkl','rb'))
preprocessor = pickle.load(open('preprocessor.pkl','rb'))

#flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('Index.html')
@app.route("/predict",methods=['POST'])
def predict():
    if request.method == 'POST':
        Rainfall = request.form['Rainfall']
        Humidity = request.form['Humidity']
        states = request.form['states']
        temperature = request.form['temperature']
        phlevel = request.form['phlevel']
        nitrogen = request.form['nitrogen']
        phosphorous = request.form['phosphorous']
        potassium = request.form['potassium']
        soiltype  = request.form['soiltype']

        features = np.array([[Year,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp,Area,Item]],dtype=object)
        transformed_features = preprocessor.transform(features)
        prediction = dtr.predict(transformed_features).reshape(1,-1)

        return render_template('index.html',prediction = prediction[0][0])

if __name__=="__main__":
    app.run(debug=True)