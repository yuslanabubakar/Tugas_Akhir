from flask import Flask, render_template, request, flash, session, redirect, url_for, jsonify
import function as fc
import kitabFunction as fcKitab
import json

app = Flask(__name__, static_url_path = "/images", static_folder = "images")
app.config["DEBUG"] = True

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/classify', methods=['GET','POST'])
def classify():
    W1M,W2M,B1M,B2M = fc.getWeights()
    igM = fc.getIGWThreshold()
    featureM = [line[:-1] for line in igM]
    matriksTargetM = fc.getMatriksTarget()

    W1K,W2K,B1K,B2K = fcKitab.getWeights()
    igK = fcKitab.getIGWThreshold()
    featureK = [line[:-1] for line in igK]
    matriksTargetK = fcKitab.getMatriksTarget()
    if request.method == 'POST':
        hadis = request.form['hadis']
        labelM,klasifikasiM = fc.testClassify(hadis,W1M,W2M,B1M,B2M,featureM,matriksTargetM)
        labelK,klasifikasiK = fcKitab.testClassify(hadis,W1K,W2K,B1K,B2K,featureK,matriksTargetK)
        result = {}
        result['klasifikasiM'] = klasifikasiM
#        result['w1'] = W1
        result['klasifikasiK'] = klasifikasiK
        return json.dumps(result)

app.run()