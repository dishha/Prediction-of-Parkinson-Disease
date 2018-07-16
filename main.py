from flask import Flask
from flask import render_template, json, request
from sklearn.externals import joblib
import json
import pandas as pd
import pickle
import numpy as np


app = Flask(__name__)

@app.route("/")
def main():
	return render_template('index.html')


@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      result = request.form
      return render_template("result.html",result = result)


@app.route('/predict', methods=['POST','GET'])
def make_prediction():
  if request.method=='POST':
    inputs = request.form
    print inputs
    prediction = model.predict(inputs)
    print prediction
    return render_template("result.html", result=result)
    #return jsonify({'prediction':list(prediction)})


if __name__ == "__main__":
	model = joblib.load('model.pkl')
	app.run()

