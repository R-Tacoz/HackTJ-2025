import os
from flask import Flask, render_template, request, jsonify, redirect
from processing.utils import exportFeatureVec, createFeatureVec
import torch
from processing.model import KeyMetAAProfileModel

app = Flask(__name__)
username = ""
test_features = None
user_profile_model = None

@app.route('/')
def home():
    global username
    print('hey')
    return render_template('index.html', username=username)

@app.route('/validate_login', methods=['POST'])
def validate_login():
    global username, user_profile_model
    username=request.form.get('username')
    
    model_path = f"./processing/data/{username}.pt"
    if os.path.exists(model_path):
        print('loaded model"')
        user_profile_model = KeyMetAAProfileModel()
        user_profile_model.load_state_dict(torch.load(f"./processing/data/{username}.pt"))
    else:
        user_profile_model = None
        
    return redirect('/')

@app.route('/logout')
def logout():
    global username
    username = ""
    return redirect('/')

@app.route('/profile/<username>')
def profile(username):
    return render_template('profile.html', username=username)

@app.route('/store', methods=['POST'])
def store():
    global test_features
    
    response = request.json.get('data')
    
    test_features = createFeatureVec(response, tensor=True)
    # exportFeatureVec(username, response)
    
    return "Worked"

@app.route('/results')
def results():
    
    if user_profile_model is not None:
        prob = user_profile_model(test_features)
    else:
        prob = 0.
        
    print("Model output:", prob)
    passed = prob > 0.5
    
    return render_template('results.html', passed=passed)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=1432, debug=True)
