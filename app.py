from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os
import traceback

app = Flask(__name__)

# Configuration
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

# Load the trained model and encoders
try:
    model = joblib.load('model/disease_model.pkl')
    disease_encoder = joblib.load('model/disease_encoder.pkl')
    gender_encoder = joblib.load('model/gender_encoder.pkl')  # This is now a LabelEncoder
    bp_map = joblib.load('model/bp_col_encoder.pkl')
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {str(e)}")
    raise

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = request.form.to_dict()
        print("Received form data:", form_data)  # Debug logging

        # Validate required fields
        required_fields = ['age', 'gender', 'blood_pressure', 'cholesterol']
        for field in required_fields:
            if not form_data.get(field):
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Prepare input data
        try:
            input_data = {
                'Age': int(form_data['age']),
                'Gender': gender_encoder.transform([form_data['gender']])[0],
                'Blood Pressure': bp_map.get(form_data['blood_pressure'], 1),
                'Cholesterol Level': bp_map.get(form_data['cholesterol'], 1),
                'Fever': 1 if form_data.get('fever') == 'Yes' else 0,
                'Cough': 1 if form_data.get('cough') == 'Yes' else 0,
                'Fatigue': 1 if form_data.get('fatigue') == 'Yes' else 0,
                'Difficulty Breathing': 1 if form_data.get('difficulty_breathing') == 'Yes' else 0
            }
        except ValueError as e:
            return jsonify({'error': f'Invalid input data: {str(e)}'}), 400

        # Convert to DataFrame with correct column order
        input_df = pd.DataFrame([input_data])[['Age', 'Gender', 'Blood Pressure', 'Cholesterol Level',
                                               'Fever', 'Cough', 'Fatigue', 'Difficulty Breathing']]

        # Make prediction
        try:
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df).max()
            disease = disease_encoder.inverse_transform(prediction)[0]

            return jsonify({
                'disease': disease,
                'probability': float(probability),
                'status': 'success'
            })

        except Exception as e:
            app.logger.error(f"Prediction failed: {str(e)}\n{traceback.format_exc()}")
            return jsonify({
                'error': 'Prediction failed',
                'details': str(e),
                'status': 'error'
            }), 500

    except Exception as e:
        app.logger.error(f"Server error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e),
            'status': 'error'
        }), 500


if __name__ == '__main__':
    os.makedirs('model', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)