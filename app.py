from flask import Flask, render_template, request
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import os
from datetime import datetime
import random


app= Flask(__name__)


@app.route('/predictions')
def predictions():
   return render_template('prediction.html')


@app.route('/')
def homepage():
   return render_template('index.html')

if __name__ == '__main__':
   app.run(debug=True, host='127.0.0.1', port=5001)


