from flask import Flask, render_template, request
import numpy as np
import joblib

# Create the Flask app instance
app = Flask(__name__)

# Load pre-trained model, scaler, and label encoders
model = joblib.load('logistic_regression_model.pkl')  # Adjust path as needed
scaler = joblib.load('scaler.pkl')  # Adjust path as needed
label_encoders = joblib.load('label_encoders.pkl')  # Adjust path as needed

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form (11 features expected by the model)
    age = int(request.form['age'])
    sex = request.form['sex']
    chest_pain_type = request.form['chestPainType']
    resting_bp = int(request.form['restingBP'])
    cholesterol = int(request.form['cholesterol'])
    fasting_bs = int(request.form['fastingBS'])
    resting_ecg = request.form['restingECG']
    max_hr = int(request.form['maxHR'])
    exercise_angina = request.form['exerciseAngina']
    oldpeak = float(request.form['oldpeak'])
    st_slope = request.form['stSlope']
    
    # Apply LabelEncoding for categorical variables using the saved label encoders
    try:
        sex = label_encoders['Sex'].transform([sex])[0]
    except ValueError:
        sex = 0  # Default value or handle error as needed

    try:
        chest_pain_type = label_encoders['ChestPainType'].transform([chest_pain_type])[0]
    except ValueError:
        chest_pain_type = 0  # Default value or handle error as needed

    try:
        resting_ecg = label_encoders['RestingECG'].transform([resting_ecg])[0]
    except ValueError:
        resting_ecg = 0  # Default value or handle error as needed

    try:
        st_slope = label_encoders['ST_Slope'].transform([st_slope])[0]
    except ValueError:
        st_slope = 0  # Default value or handle error as needed

    # Convert 'ExerciseAngina' to binary (0 or 1)
    exercise_angina = 1 if exercise_angina.lower() == 'yes' else 0

    # Prepare the feature vector (11 features)
    features = np.array([[age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]])

    # Separate numerical features for scaling
    numerical_features = features[:, [0, 3, 4, 5, 7, 9]]  # Select only numerical columns for scaling
    
    # Apply scaling using the pre-trained scaler
    numerical_features_scaled = scaler.transform(numerical_features)

    # Replace original numerical features with scaled ones in the features array
    features[:, [0, 3, 4, 5, 7, 9]] = numerical_features_scaled

    # Predict with the model
    prediction = model.predict(features)
    
    # Prepare the result
    prediction_text = 'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'
    
    return render_template('index.html', prediction_text=f'Prediction: {prediction_text}')

if __name__ == "__main__":
    app.run(debug=True)
