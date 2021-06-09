import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('CustomerCategoryFeatureMod.ft', 'wb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    age = float(request.form['age'])
    salary = float(request.form['salary'])
    finalFeatures = np.array([[age,salary]])
    prediction = model.predict(finalFeatures)

    

    return render_template('index.html', prediction_text='Given customer is a {} customer'.format(round(prediction[0][0])))


if __name__ == "__main__":
    app.run(debug=True)
