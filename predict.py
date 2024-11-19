import pickle

from flask import Flask
from flask import request
from flask import jsonify

input_file = 'model_C=0.1.bin'


with open(input_file, 'rb') as f_in: 
    dv, model = pickle.load(f_in)


app = Flask('heart_disease')

@app.route('/predict', methods=['POST'])
def predict():
    patient = request.get_json()

    X = dv.transform([patient])
    y_pred = model.predict_proba(X)[0, 1]
    heart_disease = y_pred >= 0.5

    result = {
        'heart_disease_probability': float(y_pred),
        'heart_disease': bool(heart_disease)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)