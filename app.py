from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_cors import CORS
from flask_mail import Mail, Message
import pandas as pd
import numpy as np
import tensorflow as tf
from pymongo import MongoClient
import bcrypt
import jwt
import os
import requests
from datetime import datetime, timedelta
from bson import ObjectId
from bson.json_util import dumps

app = Flask(__name__)
CORS(app)

# MongoDB Connection
client = MongoClient(
    "mongodb+srv://devopsdeveloper98:n8kTwBLCwsSJYmIA@custer-o.qj7um.mongodb.net/app-db?retryWrites=true&w=majority&appName=custer-o"
)
db = client['wellness_db']
users_collection = db['users']
logs_collection = db['logs']
admin_collection = db['admin']

# Flask-Mail Configuration
app.config['MAIL_SERVER'] = 'smtp.example.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'your-email@example.com'
app.config['MAIL_PASSWORD'] = 'your-email-password'
mail = Mail(app)

# App secret key
app.config['SECRET_KEY'] = "[uF8U_%p{xDG8R-%yH.b}eiK62iTr("
JWT_EXPIRATION_DELTA = timedelta(days=1)

# Load Updated Model (calorie_predictor_v6.keras)
model = tf.keras.models.load_model('calorie_predictor_v6.keras')

# JWT Helper Function
def generate_jwt(user_id):
    expiration_time = datetime.utcnow() + JWT_EXPIRATION_DELTA
    payload = {
        'user_id': user_id,
        'exp': expiration_time
    }
    return jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')

# JWT Token Validation Function
def decode_jwt(token):
    try:
        return jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

# Password Reset Token Generation
def generate_password_reset_token(email):
    expiration_time = datetime.utcnow() + timedelta(hours=1)
    payload = {
        'email': email,
        'exp': expiration_time
    }
    return jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')

# Fitbit API Integration
def get_fitbit_data(access_token):
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    response = requests.get('https://api.fitbit.com/1/user/-/activities/steps/date/today/1d.json', headers=headers)
    if response.status_code == 200:
        return response.json()
    return None

# Prediction Function
def predict_calories(data):
    input_data = np.array([[data['age'], data['steps'], data['weight'], data['height'], data['activity_level_encoded']]])
    prediction = model.predict(input_data)
    return float(prediction[0][0])

# BMI Calculation
def calculate_bmi(weight, height):
    height_m = height / 100  # Convert height to meters
    return round(weight / (height_m ** 2), 2)

# Calorie Goal Calculation
def calculate_calorie_goal(activity_level, weight, height, age):
    bmr = 10 * weight + 6.25 * height - 5 * age + 5  # Mifflin-St Jeor Equation for Men
    activity_factors = {
        'Sedentary': 1.2,
        'Lightly Active': 1.375,
        'Active': 1.55,
        'Very Active': 1.725
    }
    return bmr * activity_factors.get(activity_level, 1.2)

# Macronutrient Distribution
def calculate_nutrition(calories, activity_level):
    if activity_level == 'Sedentary':
        carb_ratio, protein_ratio, fat_ratio = 0.45, 0.25, 0.30
    elif activity_level == 'Lightly Active':
        carb_ratio, protein_ratio, fat_ratio = 0.50, 0.20, 0.30
    elif activity_level == 'Active':
        carb_ratio, protein_ratio, fat_ratio = 0.55, 0.20, 0.25
    else:  # Very Active
        carb_ratio, protein_ratio, fat_ratio = 0.60, 0.20, 0.20
    
    carbs = (calories * carb_ratio) / 4  # 1g of carb = 4 calories
    protein = (calories * protein_ratio) / 4  # 1g of protein = 4 calories
    fat = (calories * fat_ratio) / 9  # 1g of fat = 9 calories
    
    return {
        'carbs': round(carbs, 2),
        'protein': round(protein, 2),
        'fat': round(fat, 2)
    }

# Workout Suggestions
def suggest_workout(activity_level):
    if activity_level == 'Sedentary':
        return "Start with light stretching and a 15-minute walk."
    elif activity_level == 'Lightly Active':
        return "Consider 30 minutes of moderate-intensity cardio (e.g., brisk walking, light cycling)."
    elif activity_level == 'Active':
        return "Try 45 minutes of moderate to high-intensity exercise (e.g., jogging, cycling, swimming)."
    else:  # Very Active
        return "Focus on strength training and HIIT workouts for 60 minutes."

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/signup', methods=['POST'])
def register_user():
    try:
        data = request.json
        if not data or 'username' not in data or 'password' not in data:
            return jsonify({"error": "Invalid input data"}), 400

        # Validate password complexity
        if len(data['password']) < 8:
            return jsonify({"error": "Password must be at least 8 characters long"}), 400

        # Hash password
        hashed_password = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt())

        # Save to DB
        users_collection.insert_one({
            "username": data['username'],
            "password": hashed_password,
            "email": data.get('email', ''),
            "age": data.get('age', 0),
            "weight": data.get('weight', 0),
            "height": data.get('height', 0),
            "goals": data.get('goals', {}),
            "fitbit_access_token": data.get('fitbit_access_token', '')
        })

        return jsonify({"message": "User registered successfully!"}), 201

    except Exception as e:
        app.logger.error(f"Error registering user: {e}")
        return jsonify({"error": "An error occurred during registration"}), 500

@app.route('/api/login', methods=['POST'])
def login_user():
    data = request.json
    if not data or 'email' not in data or 'password' not in data:
        return jsonify({"error": "Invalid input data"}), 400

    user = users_collection.find_one({"email": data['email']})
    if not user:
        return jsonify({"error": "User not found"}), 404

    if bcrypt.checkpw(data['password'].encode('utf-8'), user['password']):
        token = generate_jwt(str(user['_id']))
        return jsonify({"message": "Login successful!", "token": token}), 200
    return jsonify({"error": "Incorrect password"}), 400

@app.route('/api/reset_password', methods=['POST'])
def reset_password():
    data = request.json
    email = data.get('email')
    user = users_collection.find_one({"email": email})
    if not user:
        return jsonify({"error": "User not found"}), 404

    reset_token = generate_password_reset_token(email)
    reset_link = f"https://yourapp.com/reset_password?token={reset_token}"

    msg = Message("Password Reset Request", sender="your-email@example.com", recipients=[email])
    msg.body = f"Click the link to reset your password: {reset_link}"
    mail.send(msg)

    return jsonify({"message": "Password reset link sent to your email"}), 200

@app.route('/api/update_password', methods=['POST'])
def update_password():
    data = request.json
    token = data.get('token')
    new_password = data.get('new_password')
    if not token or not new_password:
        return jsonify({"error": "Token and new password are required"}), 400

    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        email = payload.get('email')
        user = users_collection.find_one({"email": email})
        if not user:
            return jsonify({"error": "User not found"}), 404

        hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
        users_collection.update_one({"email": email}, {"$set": {"password": hashed_password}})
        return jsonify({"message": "Password updated successfully!"}), 200
    except jwt.ExpiredSignatureError:
        return jsonify({"error": "Token has expired"}), 400
    except jwt.InvalidTokenError:
        return jsonify({"error": "Invalid token"}), 400

@app.route('/api/profile', methods=['GET'])
def get_user_profile():
    token = request.headers.get('Authorization').split()
    if len(token) == 2 and token[0].lower() == 'bearer':
        bearer_token = token[1]
    else:
        bearer_token = None

    if not bearer_token:
        return jsonify({"error": "Token is missing"}), 401

    decoded_token = decode_jwt(bearer_token)
    if not decoded_token:
        return jsonify({"error": "Invalid or expired token"}), 401

    user_id = decoded_token['user_id']
    if isinstance(user_id, str):
        user_id = ObjectId(user_id)

    user = users_collection.find_one({"_id": user_id})
    if not user:
        return jsonify({"error": "User not found"}), 404

    user_profile = {
        "username": user['username'],
        "email": user.get('email', ''),
        "age": user.get('age', 0),
        "weight": user.get('weight', 0),
        "height": user.get('height', 0),
        "goals": user.get('goals', {}),
        "workouts": user.get('workout_suggestion', []),
        "meals": user.get('meals', []),
        "water_intake": user.get('water_intake', 0),
        "sleep": user.get('sleep', 0),
        "mood": user.get('mood', 0),
        "medications": user.get('medications', []),
        "allergies": user.get('allergies', []),
        "medical_conditions": user.get('medical_conditions', []),
        "emergency_contacts": user.get('emergency_contacts', []),
        "bmi": user.get('bmi', []),
        'calories': user.get('predicted_calories', []),
        "fitbit_data": get_fitbit_data(user.get('fitbit_access_token', '')),
        'nutrition': user.get('nutrition', {})
    }
    
    return jsonify({"profile": user_profile}), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    if 'age' not in data or 'steps' not in data or 'weight' not in data or 'height' not in data or 'activity_level' not in data:
        return jsonify({"error": "Missing required fields"}), 400

    activity_level_map = {'Sedentary': 0, 'Lightly Active': 1, 'Active': 2, 'Very Active': 3}
    data['activity_level_encoded'] = activity_level_map.get(data['activity_level'], 0)

    predicted_calories = predict_calories(data)
    bmi = calculate_bmi(data['weight'], data['height'])
    calorie_goal = calculate_calorie_goal(data['activity_level'], data['weight'], data['height'], data['age'])
    nutrition = calculate_nutrition(predicted_calories, data['activity_level'])
    workout = suggest_workout(data['activity_level'])

    prediction_results = {
        "predicted_calories": predicted_calories,
        "bmi": bmi,
        "calorie_goal": calorie_goal,
        "nutrition": nutrition,
        "workout_suggestion": workout
    }

    token = request.headers.get('Authorization').split()
    if len(token) == 2 and token[0].lower() == 'bearer':
        bearer_token = token[1]
    else:
        bearer_token = None

    if not bearer_token:
        return jsonify({"error": "Token is missing"}), 401

    decoded_token = decode_jwt(bearer_token)
    if not decoded_token:
        return jsonify({"error": "Invalid or expired token"}), 401

    user_id = decoded_token['user_id']
    if isinstance(user_id, str):
        user_id = ObjectId(user_id)

    users_collection.update_one(
        {"_id": user_id},
        {"$set": {
            "predicted_calories": predicted_calories,
            "bmi": bmi,
            "height": data['height'],
            "weight": data['weight'],
            "goals":calorie_goal,
            "calorie_goal": calorie_goal,
            "nutrition": nutrition,
            "workout_suggestion": workout,
            "last_prediction_date": datetime.utcnow()
        }}
    )

    return jsonify(prediction_results), 200

@app.route('/api/log_progress', methods=['POST'])
def log_progress():
    data = request.json
    user_id = data.get('user_id')
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    log_data = {
        'user_id': user_id,
        'date': pd.Timestamp.now(),
        'steps': data.get('steps'),
        'calories_consumed': data.get('calories_consumed'),
        'weight': data.get('weight')
    }
    logs_collection.insert_one(log_data)
    return jsonify({"message": "Progress logged successfully!"}), 201

@app.route('/api/admin/dashboard', methods=['GET'])
def admin_dashboard():
    # Fetch all users and their progress
    users = list(users_collection.find({}, {"username": 1, "email": 1, "last_prediction_date": 1}))
    logs = list(logs_collection.find({}, {"user_id": 1, "date": 1, "steps": 1, "calories_consumed": 1, "weight": 1}))
    
    return jsonify({
        "users": users,
        "logs": logs
    }), 200

if __name__ == '__main__':
    app.run(debug=True)