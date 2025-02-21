from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_cors import CORS
import pandas as pd
import numpy as np
import tensorflow as tf
from pymongo import MongoClient
import bcrypt
import jwt
import os
from datetime import datetime, timedelta
from bson import ObjectId

app = Flask(__name__)
CORS(app)

# MongoDB Connection
client = MongoClient(
    "mongodb+srv://devopsdeveloper98:n8kTwBLCwsSJYmIA@custer-o.qj7um.mongodb.net/app-db?retryWrites=true&w=majority&appName=custer-o"
)

db = client['wellness_db']
users_collection = db['users']

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

# Prediction Function (Updated to use the new model with activity level encoded)
def predict_calories(data):
    input_data = np.array([[data['age'], data['steps'], data['weight'], data['height'], data['activity_level_encoded']]])
    prediction = model.predict(input_data)
    return float(prediction[0][0])

# BMI Calculation
def calculate_bmi(weight, height):
    height_m = height / 100  # Convert height to meters
    return round(weight / (height_m ** 2), 2)

# Calorie Goal Based on Activity Level
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
            "height": data.get('height', 0)
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

@app.route('/api/profile', methods=['GET'])
def get_user_profile():
    # Get the token from the request header
    token = request.headers.get('Authorization').split()
    if len(token) == 2 and token[0].lower() == 'bearer':
        bearer_token = token[1]
    else:
        bearer_token = None

    if not bearer_token:
        return jsonify({"error": "Token is missing"}), 401

    # Decode the token and extract user ID
    decoded_token = decode_jwt(bearer_token)
    print(decoded_token)
    if not decoded_token:
        return jsonify({"error": "Invalid or expired token"}), 401

    user_id = decoded_token['user_id']

    # Convert string user_id to ObjectId if needed
    if isinstance(user_id, str):
        user_id = ObjectId(user_id)


    # Fetch user profile from DB using the user ID
    user = users_collection.find_one({"_id": user_id})
    print(user)
    if not user:
        return jsonify({"error": "User not found"}), 404

    # Return user profile details
    user_profile = {
        "username": user['username'],
        "email": user.get('email', ''),
        "age": user.get('age', 0),
        "weight": user.get('weight', 0),
        "height": user.get('height', 0)
    }
    
    return jsonify({"profile": user_profile}), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    if 'age' not in data or 'steps' not in data or 'weight' not in data or 'height' not in data or 'activity_level' not in data:
        return jsonify({"error": "Missing required fields"}), 400
    age = float(data.get('age', 0))  # Use default value of 0 if not provided
    steps = float(data.get('steps', 0))
    weight = float(data.get('weight', 0))
    height = float(data.get('height', 0))
    activity_level = data.get('activity_level', '')

    # Encode activity level (Assumes the activity level is a string and needs to be encoded)
    activity_level_map = {'Sedentary': 0, 'Lightly Active': 1, 'Active': 2, 'Very Active': 3}
    data['activity_level_encoded'] = activity_level_map.get(data['activity_level'], 0)

    # Predict the calories
    predicted_calories = predict_calories(data)
    
    # Calculate BMI
    bmi = calculate_bmi(data['weight'], data['height'])
    
    # Calculate Calorie Goal
    calorie_goal = calculate_calorie_goal(data['activity_level'], data['weight'], data['height'], data['age'])
    
    # Calculate Macronutrient Distribution
    nutrition = calculate_nutrition(predicted_calories, data['activity_level'])
    
    # Workout suggestion based on activity level
    workout = suggest_workout(data['activity_level'])
    
    # Prepare the prediction results data
    prediction_results = {
        "predicted_calories": predicted_calories,
        "bmi": bmi,
        "calorie_goal": calorie_goal,
        "nutrition": nutrition,
        "workout_suggestion": workout
    }

    # Get the token from the request headers
    token = request.headers.get('Authorization').split()
    if len(token) == 2 and token[0].lower() == 'bearer':
        bearer_token = token[1]
    else:
        bearer_token = None

    if not bearer_token:
        return jsonify({"error": "Token is missing"}), 401

    # Decode the token and extract user ID
    decoded_token = decode_jwt(bearer_token)
    if not decoded_token:
        return jsonify({"error": "Invalid or expired token"}), 401

    user_id = decoded_token['user_id']

    # Convert string user_id to ObjectId if needed
    if isinstance(user_id, str):
        user_id = ObjectId(user_id)

    # Update the user's profile in the database with the prediction results
    users_collection.update_one(
        {"_id": user_id},
        {"$set": {
            "predicted_calories": predicted_calories,
            "bmi": bmi,
            "calorie_goal": calorie_goal,
            "nutrition": nutrition,
            "workout_suggestion": workout,
            "last_prediction_date": datetime.utcnow()  # Store the date of last prediction
        }}
    )

    return jsonify({
        "calories": predicted_calories,
        "bmi": bmi,
        "calorie_goal": calorie_goal,
        "nutrition": nutrition,
        "workout_suggestion": workout
    }), 200

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
    db.logs.insert_one(log_data)
    return jsonify({"message": "Progress logged successfully!"}), 201

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True)
