# AI-Powered Disease Prediction System
An intelligent web application that predicts potential diseases based on user-reported symptoms and health metrics using machine learning.

## Features
- 🩺 **Symptom Analysis**: Input multiple symptoms for comprehensive assessment
- 🤖 **Machine Learning**: Random Forest classifier trained on medical data
- 📊 **Probability Scores**: See prediction confidence levels
- 🌐 **Web Interface**: Clean, responsive design with Tailwind CSS
- 🚀 **Flask Backend**: Lightweight and efficient Python web framework

## Technology Stack

**Frontend**:
- Tailwind CSS
- Alpine.js for reactive components
- HTML5, JavaScript

**Backend**:
- Python Flask
- Scikit-learn (Random Forest Classifier)
- Pandas for data processing

**Machine Learning**:
- Random Forest algorithm
- Label Encoding for categorical data
- Standard Scaling for numerical features

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/tashitams/disease-predictor.git
   cd disease-predictor

2. Create and activate virtual environment::
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt

4. Train the model (optional - includes sample dataset):
   ```bash
   python train_model.py

5. Run the application:
   ```bash
   python app.py

6. Access the application at: http://localhost:5000

## Dataset
The model is trained on a synthetic dataset containing:
- 🩺 Common symptoms (Fever, Cough, Fatigue, etc.)
- 🤖 Patient demographics (Age, Gender)
- 📊 Health metrics (Blood Pressure, Cholesterol)
- 🤒 Disease outcomes

 ## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create your feature branch (git checkout -b feature/your-feature)
3. Commit your changes (git commit -m 'Add some feature')
4. Push to the branch (git push origin feature/your-feature)
5. Open a Pull Request

## Disclaimer
⚠️ Important: This application is for educational purposes only and should not be used for actual medical diagnosis. Always consult a healthcare professional for medical advice.

## Future Enhancements
- Add more diseases and symptoms
- Implement user accounts
- Add historical prediction tracking
- Integrate with medical APIs
  
