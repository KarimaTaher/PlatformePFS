from flask import Flask,render_template
app= Flask(__name__)

@app.route('/')
def homepage():
   return render_template('index.html')

@app.route('/graphs')
def show_graphs():
    return render_template('graphs.html', )