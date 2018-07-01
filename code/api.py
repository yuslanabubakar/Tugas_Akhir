from flask import Flask, render_template, request, flash, session, redirect, url_for, jsonify
import function as fc
import json

app = Flask(__name__, static_url_path = "/images", static_folder = "images")
app.config["DEBUG"] = True

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/classify', methods=['GET','POST'])
def classify():
    W1,W2,B1,B2 = fc.getWeights()
    ig = fc.getIGWThreshold()
    feature = [line[:-1] for line in ig]
    matriksTarget = fc.getMatriksTarget()
    if request.method == 'POST':
        hadis = request.form['hadis']
        label,klasifikasi = fc.testClassify(hadis,W1,W2,B1,B2,feature,matriksTarget)
        result = {}
        result['label'] = label
#        result['w1'] = W1
        result['klasifikasi'] = klasifikasi
        return json.dumps(result)

app.run()