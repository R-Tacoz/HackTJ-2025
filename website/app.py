from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', text="What you have typed so far")

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/key_press', methods=['POST'])
def key_press():
    alpha="abcdefghijklmnopqrstuvwxyz"
    alpha+=alpha.upper()+" "
    key = request.json.get('key')
    response = {
        'valid': key in alpha,
        'key': key
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)