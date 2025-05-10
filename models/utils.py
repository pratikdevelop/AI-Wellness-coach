import numpy as np
import bcrypt
import jwt
from datetime import datetime, timedelta
import threading
import requests
from werkzeug.utils import secure_filename
import uuid
import os
from bson import ObjectId

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def generate_jwt(app, user_id, is_admin=False):
    expiration_time = datetime.utcnow() + timedelta(seconds=int(app.config['JWT_EXPIRATION_DELTA']))
    payload = {'user_id': user_id, 'is_admin': is_admin, 'exp': expiration_time}
    return jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')

def decode_jwt(app, token):
    try:
        return jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None

def generate_password_reset_token(app, email):
    expiration_time = datetime.utcnow() + timedelta(hours=1)
    payload = {'email': email, 'exp': expiration_time}
    return jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')

def send_async_email(app, msg):
    with app.app_context():
        app.mail.send(msg)

def send_reset_email(app, email, token):
    from flask_mail import Message
    msg = Message("Password Reset Request", sender=app.config['MAIL_USERNAME'], recipients=[email])
    msg.body = f"Click this link to reset your password: http://localhost:5000/reset_password/{token}"
    threading.Thread(target=send_async_email, args=(app, msg)).start()

def get_fitbit_data(access_token):
    headers = {'Authorization': f'Bearer {access_token}'}
    fitbit_data = {}
    steps_response = requests.get('https://api.fitbit.com/1/user/-/activities/steps/date/today/1d.json', headers=headers)
    if steps_response.status_code == 200:
        fitbit_data['steps'] = steps_response.json().get('activities-steps', [{}])[0].get('value', 0)
    sleep_response = requests.get('https://api.fitbit.com/1.2/user/-/sleep/date/today.json', headers=headers)
    if sleep_response.status_code == 200:
        sleep_data = sleep_response.json().get('sleep', [{}])[0]
        fitbit_data['sleep_duration'] = sleep_data.get('duration', 0) / 3600000
    heart_response = requests.get('https://api.fitbit.com/1/user/-/activities/heart/date/today/1d.json', headers=headers)
    if heart_response.status_code == 200:
        heart_data = heart_response.json().get('activities-heart', [{}])[0].get('value', {})
        fitbit_data['heart_rate'] = heart_data.get('restingHeartRate', 0)
    return fitbit_data if fitbit_data else None

def predict_calories(app, data):
    input_data = np.array([[data['age'], data['steps'], data['weight'], data['height'], 
                            data['activity_level_encoded'], data['gender_encoded'], 
                            data['heart_rate'], data['sleep_hours'], data['stress_level'],
                            data['mood_encoded']]])  # Added mood_encoded
    input_data_scaled = app.scaler.transform(input_data)
    prediction = app.model.predict(input_data_scaled, verbose=0)
    return float(prediction[0][0])
def calculate_bmi(weight, height):
    height_m = height / 100
    return round(weight / (height_m ** 2), 2)

def calculate_calorie_goal(activity_level, weight, height, age, gender, sleep_hours, stress_level):
    if gender == 'M':
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    activity_factors = {'Sedentary': 1.2, 'Lightly Active': 1.375, 'Active': 1.55, 'Very Active': 1.725}
    base_tdee = bmr * activity_factors.get(activity_level, 1.2)
    sleep_adjustment = 1 - (0.02 * (8 - sleep_hours))
    stress_adjustment = 1 + (0.03 * (stress_level / 10))
    return round(base_tdee * sleep_adjustment * stress_adjustment, 2)

def calculate_nutrition(calories, activity_level):
    ratios = {
        'Sedentary': (0.45, 0.25, 0.30),
        'Lightly Active': (0.50, 0.20, 0.30),
        'Active': (0.55, 0.20, 0.25),
        'Very Active': (0.60, 0.20, 0.20)
    }
    carb_ratio, protein_ratio, fat_ratio = ratios.get(activity_level, (0.45, 0.25, 0.30))
    return {
        'carbs': round((calories * carb_ratio) / 4, 2),
        'protein': round((calories * protein_ratio) / 4, 2),
        'fat': round((calories * fat_ratio) / 9, 2)
    }

def suggest_workout(app, state):
    """Use RL agent to suggest a workout based on the current state."""
    workout = app.rl_agent.get_action(state)
    workouts = {
        'Sedentary': "Start with light stretching and a 15-minute walk.",
        'Lightly Active': "Consider 30 minutes of moderate-intensity cardio.",
        'Active': "Try 45 minutes of moderate to high-intensity exercise.",
        'Very Active': "Focus on strength training and HIIT for 60 minutes."
    }
    return workouts[workout], workout  # Return description and action key