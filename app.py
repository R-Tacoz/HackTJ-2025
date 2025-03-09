
from flask import Flask, render_template, request, jsonify, redirect
from processing.utils import exportFeatureVec


app = Flask(__name__)
username=""

@app.route('/')
def home():
    global username
    return render_template('index.html', username=username)

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/validate_login', methods=['POST'])
def validate_login():
    global username
    username=request.form.get('username')
    return redirect('/')

@app.route('/logout')
def logout():
    global username
    username=""
    return redirect('/')

@app.route('/profile/<username>')
def profile(username):
    return render_template('profile.html', username=username)

@app.route('/store', methods=['POST'])
def store():
    response = request.json.get('data')
    exportFeatureVec(response)
    return "Worked"

@app.route('/results')
def results():
    return render_template('results.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=1432, debug=True)
