from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('diabetes_model.pkl')

# Define feature names
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form
        input_data = [float(data[feature]) for feature in feature_names]
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data], columns=feature_names)
        
        # Predict probability without scaling
        probability = model.predict_proba(input_df)[:, 1][0]
        probability_percentage = probability * 100
        
        # Render result page
        return render_template('result.html', probability=probability_percentage)
    
    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
