import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import os

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def my_form():
    return render_template('Home.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text1 = request.form['text1']
    text2 = request.form['text2']
    text3 = request.form['text3']
    #input = str(type(text1))
    result = model.predict([[text1,text2,text3]])
    return jsonify({'answer': result[0]}) #results to a json showing the output of the model

if __name__=="__main__":
    app.run(host=os.getenv('IP', '0.0.0.0'),
            port=int(os.getenv('PORT', 4444)))
