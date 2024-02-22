#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    features_array = np.array([features])
    prediction = model.predict(features_array)

    if prediction[0] == 1:
        result = "Safe to drink"
    else:
        result = "Not safe to drink"

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)

