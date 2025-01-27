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

app = Flask(__name__)
CORS(app)

# MongoDB Connection
client = MongoClient(
    "mongodb+srv://game:FiGFqvOmjjclauba@cluster0.rnzw1.mongodb.net/app-db?retryWrites=true&w=majority&appName=Cluster0"
)
db = client['wellness_db']
users_collection = db['users']

model = tf.keras.models.load_model('calorie_predictor_v3.h5')

from flask_jwt_extended import jwt_required

# Prediction Function
def predict_calories(data):
    input_data = np.array([[data['age'], data['steps'], data['weight'], data['height']]])
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

# Routes
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/api/register', methods=['POST'])
def register_user():
    data = request.json
    if not data:
        return jsonify({"error": "Invalid input data"}), 400
    users_collection.insert_one(data)
    return jsonify({"message": "User registered successfully!"}), 201

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    if 'age' not in data or 'steps' not in data or 'weight' not in data or 'height' not in data or 'activity_level' not in data:
        return jsonify({"error": "Missing required fields"}), 400
    
    # Predict calories
    result = predict_calories(data)
    
    # Calculate BMI
    bmi = calculate_bmi(data['weight'], data['height'])
    
    # Calculate daily calorie goal
    calorie_goal = calculate_calorie_goal(data['activity_level'], data['weight'], data['height'], data['age'])
    
    return jsonify({
        "calories": result,
        "bmi": bmi,
        "calorie_goal": calorie_goal
    }), 200

if __name__ == '__main__':from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import tensorflow as tf
from pymongo import MongoClient

app = Flask(__name__)
CORS(app)

# MongoDB Connection
client = MongoClient(
    "mongodb+srv://game:FiGFqvOmjjclauba@cluster0.rnzw1.mongodb.net/app-db?retryWrites=true&w=majority&appName=Cluster0"
)
db = client['wellness_db']
users_collection = db['users']
app.config['SECRET_KEY'] = os.urandom(24)
JWT_EXPIRATION_DELTA = timedelta(days=1)
model = tf.keras.models.load_model('calorie_predictor_v3.h5')


def generate_jwt(user_id):
    """
    Function to generate a JWT for a user
    """
    print(user_id)
    expiration_time = datetime.utcnow() + JWT_EXPIRATION_DELTA
    payload = {
        'user_id': user_id,
        'exp': expiration_time
    }
    token = jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')
    return token
# Prediction Function
def predict_calories(data):
    input_data = np.array([[data['age'], data['steps'], data['weight'], data['height']]])
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
        'carbs': round(carb, 2),
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

@app.route('/api/signup', methods=['POST'])
def register_user():
    data = request.json
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({"error": "Invalid input data"}), 400

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
@app.route('/api/login', methods=['POST'])
def login_user():
    data = request.json
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({"error": "Invalid input data"}), 400

    # Find the user in MongoDB
    user = users_collection.find_one({"username": data['username']})
    if not user:
        return jsonify({"error": "User not found"}), 404

    # Check the password
    if bcrypt.checkpw(data['password'].encode('utf-8'), user['password']):
        # Generate JWT for the authenticated user
        token = generate_jwt(str(user['_id']))
        return jsonify({"message": "Login successful!", "token": token}), 200
    else:
        return jsonify({"error": "Incorrect password"}), 400


@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/login')
def login():
    return render_template('login.html')


# Progress Logging (optional)
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
    
    # Save the log to the database
    db.logs.insert_one(log_data)
    return jsonify({"message": "Progress logged successfully!"}), 201

# Prediction and Calorie Calculation Route
@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    if 'age' not in data or 'steps' not in data or 'weight' not in data or 'height' not in data or 'activity_level' not in data:
        return jsonify({"error": "Missing required fields"}), 400
    
    # Predict calories
    predicted_calories = predict_calories(data)
    
    # Calculate BMI
    bmi = calculate_bmi(data['weight'], data['height'])
    
    # Calculate daily calorie goal
    calorie_goal = calculate_calorie_goal(data['activity_level'], data['weight'], data['height'], data['age'])
    
    # Nutritional recommendations
    nutrition = calculate_nutrition(predicted_calories, data['activity_level'])
    
    # Workout suggestions
    workout = suggest_workout(data['activity_level'])
    
    return jsonify({
        "calories": predicted_calories,
        "bmi": bmi,
        "calorie_goal": calorie_goal,
        "nutrition": nutrition,
        "workout_suggestion": workout  # Add workout suggestion
    }), 200

# Index Route
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/dashboard')
# @jwt_required()  # Ensures only logged-in users can access this page
def dashboard():    
    return render_template('dashboard.html', user=user_data)


@app.route('/api/user-data')
def get_user_data():
    user_data = {
        "username": "JohnDoe",
        "calories": 2000,
        "bmi": 22.5,
        "calorie_goal": 2500,
        "steps_today": 7500,
        "activity_level": "Moderate"
    }
    return jsonify(user_data)
if __name__ == '__main__':
    app.run(debug=True)
