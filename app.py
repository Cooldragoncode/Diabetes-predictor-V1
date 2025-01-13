import os
from flask import Flask, request, render_template
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('diabetes_model.pkl')

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')  # Serves the main HTML page

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input data from the form
        data = [float(x) for x in request.form.values()]
        
        # Convert data to a numpy array and predict the probability
        probability = model.predict_proba([np.array(data)])[0][1]  # Probability for the positive class
        
        # Prepare a result message with the probability
        result = f"The probability of diabetes is {probability * 100:.2f}%"
        
        # Pass the result to the result.html template
        return render_template('result.html', prediction_text=result)
    except Exception as e:
        return render_template('result.html', prediction_text=f"Error: {e}")

# Run the app
if __name__ == '__main__':
    # Use the PORT environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
