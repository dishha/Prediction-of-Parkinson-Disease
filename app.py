from flask import render_template, json, request, flash
from flask import Flask
import pandas as pd
import numpy as np
import pickle
import json

import warnings
warnings.filterwarnings('ignore')



app = Flask(__name__)

@app.route("/")
def main():
	return render_template('index.html')


@app.route('/result',methods = ['POST', 'GET'])
def result():
    result = 'TEST'
    if request.method == 'POST':
        result = request.form
        return render_template("result.html",result = result)
    return render_template("result.html",result = result)


@app.route('/predict', methods=['POST','GET'])
def make_prediction():
    if request.method=='POST':
        Fo = float(str(request.form.get('MDVP:Fo(Hz)')))
        Fhi = float(str(request.form.get('MDVP:Fhi(Hz)')))
        Flo = float(str(request.form.get('MDVP:Flo(Hz)')))
        Jitter = float(str(request.form.get('MDVP:Jitter(Abs)')))
        rap = float(str(request.form.get('MDVP:RAP')))
        ppq = float(str(request.form.get('MDVP:PPQ')))
        ddp = float(str(request.form.get('Jitter:DDP')))
        shimmer = float(str(request.form.get('MDVP:Shimmer')))
        dB = float(str(request.form.get('MDVP:Shimmer(dB)')))
        apq3 = float(str(request.form.get('Shimmer:APQ3')))
        apq5 = float(str(request.form.get('Shimmer:APQ5')))
        apq = float(str(request.form.get('MDVP:APQ')))
        dda = float(str(request.form.get('Shimmer:DDA')))
        nhr = float(str(request.form.get('NHR')))
        hnr = float(str(request.form.get('HNR')))
        rpde = float(str(request.form.get('RPDE')))
        dfa = float(str(request.form.get('DFA')))
        spread1 = float(str(request.form.get('spread1')))
        spread2 = float(str(request.form.get('spread2')))
        d2 = float(str(request.form.get('D2')))
        ppe = float(str(request.form.get('PPE')))

        testData = np.array([Fo,Fhi,Flo,Jitter,rap,ppq,ddp,shimmer,dB,apq3,apq5,apq,dda,nhr,hnr,rpde,dfa,spread1,spread2,d2,ppe]).reshape(1,21)
        #testData = [Fo,Fhi,Flo,Jitter,rap,ppq,ddp,shimmer,dB,apq3,apq5,apq,dda,nhr,hnr,rpde,dfa,spread1,spread2,d2,ppe]
        #print (len(testData))
        
        model = pickle.load(open('utilities/mlModel.sav', 'rb'))
        class_predicted = model.predict(testData)[0]
        


        #model = joblib.load('utilities/model.pkl')
        #class_predicted = model.predict(testData)[0]
        result = "Predicted Class: " + str(class_predicted)
        #print result
        return render_template("result.html", result=result)
        #return render_template("index.html")
    #return jsonify({'prediction':list(prediction)})
    return render_template("index.html")


if __name__ == "__main__":
	app.run()

