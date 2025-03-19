from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_mail import Mail, Message
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

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# MongoDB Connection
MONGO_URL = os.getenv("MONGO_URL")
client = MongoClient(MONGO_URL)
db = client['wellness_db']
users_collection = db['users']
logs_collection = db['logs']
admin_collection = db['admin']  # Note: Not fully utilized yet, included for future admin features

# Flask-Mail Configuration
app.config['MAIL_SERVER'] = os.getenv("MAIL_SERVER")
app.config['MAIL_PORT'] = int(os.getenv("MAIL_PORT", 465))  # Default to 465 for SSL
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv("MAIL_PASSWORD")
mail = Mail(app)

# App Secret Key and JWT Expiration
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY")
JWT_EXPIRATION_DELTA = timedelta(days=1)

# Load the TensorFlow Model
model = tf.keras.models.load_model('calorie_predictor_v6.keras')

# **JWT Helper Functions**
def generate_jwt(user_id, is_admin):
    """Generate a JWT token with user_id and is_admin flag."""
    expiration_time = datetime.utcnow() + JWT_EXPIRATION_DELTA
    payload = {
        'user_id': user_id,
        'is_admin': is_admin,
        'exp': expiration_time
    }
    return jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')

def decode_jwt(token):
    """Decode and validate a JWT token."""
    try:
        return jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None

# **Password Reset Token Generation**
def generate_password_reset_token(email):
    expiration_time = datetime.utcnow() + timedelta(hours=1)
    payload = {'email': email, 'exp': expiration_time}
    return jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')

# **Fitbit API Integration**
def get_fitbit_data(access_token):
    """Fetch step data from Fitbit API."""
    headers = {'Authorization': f'Bearer {access_token}'}
    response = requests.get('https://api.fitbit.com/1/user/-/activities/steps/date/today/1d.json', headers=headers)
    return response.json() if response.status_code == 200 else None

# **Prediction and Calculation Functions**
def predict_calories(data):
    """Predict calories burned using the TensorFlow model."""
    input_data = np.array([[data['age'], data['steps'], data['weight'], data['height'], data['activity_level_encoded']]])
    prediction = model.predict(input_data)
    return float(prediction[0][0])

def calculate_bmi(weight, height):
    """Calculate BMI from weight (kg) and height (cm)."""
    height_m = height / 100  # Convert to meters
    return round(weight / (height_m ** 2), 2)

def calculate_calorie_goal(activity_level, weight, height, age):
    """Calculate total daily calorie goal using Mifflin-St Jeor equation."""
    bmr = 10 * weight + 6.25 * height - 5 * age + 5  # For men; adjust for women if needed
    activity_factors = {
        'Sedentary': 1.2,
        'Lightly Active': 1.375,
        'Active': 1.55,
        'Very Active': 1.725
    }
    return round(bmr * activity_factors.get(activity_level, 1.2), 2)

def calculate_nutrition(calories, activity_level):
    """Calculate macronutrient distribution based on total calories."""
    if activity_level == 'Sedentary':
        carb_ratio, protein_ratio, fat_ratio = 0.45, 0.25, 0.30
    elif activity_level == 'Lightly Active':
        carb_ratio, protein_ratio, fat_ratio = 0.50, 0.20, 0.30
    elif activity_level == 'Active':
        carb_ratio, protein_ratio, fat_ratio = 0.55, 0.20, 0.25
    else:  # Very Active
        carb_ratio, protein_ratio, fat_ratio = 0.60, 0.20, 0.20
    
    carbs = (calories * carb_ratio) / 4  # 4 cal/g for carbs
    protein = (calories * protein_ratio) / 4  # 4 cal/g for protein
    fat = (calories * fat_ratio) / 9  # 9 cal/g for fat
    
    return {
        'carbs': round(carbs, 2),
        'protein': round(protein, 2),
        'fat': round(fat, 2)
    }

def suggest_workout(activity_level):
    """Provide static workout suggestions based on activity level."""
    if activity_level == 'Sedentary':
        return "Start with light stretching and a 15-minute walk."
    elif activity_level == 'Lightly Active':
        return "Consider 30 minutes of moderate-intensity cardio (e.g., brisk walking, light cycling)."
    elif activity_level == 'Active':
        return "Try 45 minutes of moderate to high-intensity exercise (e.g., jogging, cycling, swimming)."
    else:  # Very Active
        return "Focus on strength training and HIIT workouts for 60 minutes."

# **Routes**
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/signup', methods=['POST'])
def register_user():
    """Register a new user."""
    data = request.json
    if not all(k in data for k in ['username', 'password', 'email']):
        return jsonify({"error": "Username, password, and email are required"}), 400
    
    if len(data['password']) < 8:
        return jsonify({"error": "Password must be at least 8 characters long"}), 400
    
    if users_collection.find_one({"email": data['email']}):
        return jsonify({"error": "Email already registered"}), 400
    
    hashed_password = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt())
    user_data = {
        "username": data['username'],
        "password": hashed_password,
        "email": data['email'],
        "age": data.get('age', 0),
        "weight": data.get('weight', 0),
        "height": data.get('height', 0),
        "fitbit_access_token": data.get('fitbit_access_token', ''),
        "is_admin": data.get('is_admin', False)  # Default to False unless specified
    }
    users_collection.insert_one(user_data)
    return jsonify({"message": "User registered successfully!"}), 201

@app.route('/api/login', methods=['POST'])
def login_user():
    """Log in a user and return a JWT token."""
    data = request.json
    if not all(k in data for k in ['email', 'password']):
        return jsonify({"error": "Email and password are required"}), 400
    
    user = users_collection.find_one({"email": data['email']})
    if not user or not bcrypt.checkpw(data['password'].encode('utf-8'), user['password']):
        return jsonify({"error": "Invalid email or password"}), 401
    
    token = generate_jwt(str(user['_id']), user.get('is_admin', False))
    return jsonify({"message": "Login successful!", "token": token}), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict calories burned and calculate health metrics."""
    data = request.json
    required_fields = ['age', 'steps', 'weight', 'height', 'activity_level']
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400
    
    # Encode activity level
    activity_level_map = {'Sedentary': 0, 'Lightly Active': 1, 'Active': 2, 'Very Active': 3}
    data['activity_level_encoded'] = activity_level_map.get(data['activity_level'], 0)
    
    # Perform predictions and calculations
    calories_burned = predict_calories(data)
    bmi = calculate_bmi(data['weight'], data['height'])
    calorie_goal = calculate_calorie_goal(data['activity_level'], data['weight'], data['height'], data['age'])
    nutrition = calculate_nutrition(calorie_goal, data['activity_level'])  # Use calorie_goal, not calories_burned
    workout = suggest_workout(data['activity_level'])
    
    # Prepare response
    prediction_results = {
        "calories_burned": calories_burned,
        "bmi": bmi,
        "calorie_intake_goal": calorie_goal,
        "nutrition": nutrition,
        "workout_suggestion": workout
    }
    
    # Authenticate user and update profile
    token = request.headers.get('Authorization', '').split()
    if len(token) != 2 or token[0].lower() != 'bearer':
        return jsonify({"error": "Token is missing"}), 401
    
    decoded_token = decode_jwt(token[1])
    if not decoded_token:
        return jsonify({"error": "Invalid or expired token"}), 401
    
    user_id = ObjectId(decoded_token['user_id'])
    users_collection.update_one(
        {"_id": user_id},
        {"$set": {
            "calories_burned": calories_burned,
            "bmi": bmi,
            "height": data['height'],
            "weight": data['weight'],
            "calorie_goal": calorie_goal,
            "nutrition": nutrition,
            "workout_suggestion": workout,
            "last_prediction_date": datetime.utcnow(),
            "age": data['age'],
            "activity_level": data['activity_level'],
            "steps": data['steps']
        }}
    )
    
    return jsonify(prediction_results), 200

@app.route('/api/profile', methods=['GET'])
def get_user_profile():
    """Retrieve the user's profile data."""
    token = request.headers.get('Authorization', '').split()
    if len(token) != 2 or token[0].lower() != 'bearer':
        return jsonify({"error": "Token is missing"}), 401
    
    decoded_token = decode_jwt(token[1])
    if not decoded_token:
        return jsonify({"error": "Invalid or expired token"}), 401
    
    user_id = ObjectId(decoded_token['user_id'])
    user = users_collection.find_one({"_id": user_id})
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    user_profile = {
        "username": user['username'],
        "email": user.get('email', ''),
        "age": user.get('age', 0),
        "weight": user.get('weight', 0),
        "height": user.get('height', 0),
        "calorie_goal": user.get('calorie_goal', 0),
        "workouts": user.get('workout_suggestion', ''),
        "meals": user.get('meals', []),
        "water_intake": user.get('water_intake', 0),
        "sleep": user.get('sleep', 0),
        "mood": user.get('mood', ''),
        "medications": user.get('medications', []),
        "allergies": user.get('allergies', []),
        "medical_conditions": user.get('medical_conditions', []),
        "emergency_contacts": user.get('emergency_contacts', []),
        "bmi": user.get('bmi', 0),
        "calories_burned": user.get('calories_burned', 0),
        "fitbit_data": get_fitbit_data(user.get('fitbit_access_token', '')),
        "nutrition": user.get('nutrition', {}),
        "activity_level": user.get('activity_level', ''),
        "steps": user.get('steps', 0)
    }
    return jsonify(user_profile), 200

@app.route('/api/log_progress', methods=['POST'])
def log_progress():
    """Log user progress with enhanced fields."""
    data = request.json
    user_id = data.get('user_id')
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400
    
    log_data = {
        'user_id': user_id,
        'date': datetime.utcnow(),
        'steps': data.get('steps'),
        'calories_consumed': data.get('calories_consumed'),
        'carbs_consumed': data.get('carbs_consumed'),
        'protein_consumed': data.get('protein_consumed'),
        'fat_consumed': data.get('fat_consumed'),
        'weight': data.get('weight'),
        'water_intake': data.get('water_intake'),
        'sleep_hours': data.get('sleep_hours'),
        'mood': data.get('mood')
    }
    logs_collection.insert_one(log_data)
    
    # Update user's latest values in profile
    update_fields = {}
    for field, value in [
        ('weight', data.get('weight')),
        ('water_intake', data.get('water_intake')),
        ('sleep', data.get('sleep_hours')),
        ('mood', data.get('mood'))
    ]:
        if value is not None:
            update_fields[field] = value
    
    if update_fields:
        users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": update_fields}
        )
    
    return jsonify({"message": "Progress logged successfully!"}), 201

@app.route('/api/admin/dashboard', methods=['GET'])
def admin_dashboard():
    """Admin dashboard with authentication."""
    token = request.headers.get('Authorization', '').split()
    if len(token) != 2 or token[0].lower() != 'bearer':
        return jsonify({"error": "Token is missing"}), 401
    
    decoded_token = decode_jwt(token[1])
    if not decoded_token or not decoded_token.get('is_admin', False):
        return jsonify({"error": "Admin access required"}), 403
    
    users = list(users_collection.find({}, {"username": 1, "email": 1, "last_prediction_date": 1}))
    logs = list(logs_collection.find({}, {"user_id": 1, "date": 1, "steps": 1, "calories_consumed": 1, "weight": 1}))
    
    return jsonify({
        "users": users,
        "logs": logs
    }), 200

# Placeholder for Password Reset Routes (unchanged from original)
@app.route('/api/reset_password_request', methods=['POST'])
def reset_password_request():
    data = request.json
    user = users_collection.find_one({"email": data['email']})
    if not user:
        return jsonify({"error": "Email not found"}), 404
    
    token = generate_password_reset_token(data['email'])
    msg = Message("Password Reset Request", sender=app.config['MAIL_USERNAME'], recipients=[data['email']])
    msg.body = f"Click this link to reset your password: http://localhost:5000/reset_password/{token}"
    mail.send(msg)
    return jsonify({"message": "Password reset email sent"}), 200

@app.route('/api/reset_password/<token>', methods=['POST'])
def reset_password(token):
    decoded_token = decode_jwt(token)
    if not decoded_token:
        return jsonify({"error": "Invalid or expired token"}), 401
    
    data = request.json
    hashed_password = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt())
    users_collection.update_one(
        {"email": decoded_token['email']},
        {"$set": {"password": hashed_password}}
    )
    return jsonify({"message": "Password reset successfully"}), 200

if __name__ == '__main__':
    app.run(debug=True)