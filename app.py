
from flask import Flask, render_template, request, jsonify
from processing.utils import createFeatureVec


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', text="What you have typed so far")

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/validate_login', methods=['POST'])
def validate_login():
    username=request.form.get('username')
    return render_template('index.html', username=username)

@app.route('/profile/<username>')
def profile(username):
    return render_template('profile.html', username=username)

# @app.route('/key_press', methods=['POST'])
# def key_press():
#     alpha="abcdefghijklmnopqrstuvwxyz"
#     alpha+=alpha.upper()+" "
#     key = request.json.get('key')
#     response = {
#         'valid': key in alpha,
#         'key': key
#     }
#     return jsonify(response)

@app.route('/store', methods=['POST'])
def store():
    strokedata=request.json.get('data')
    
    print("stroke that thang\n", strokedata)
    return "Worked"

@app.route('/results')
def results():
    return render_template('results.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=1432, debug=True)
