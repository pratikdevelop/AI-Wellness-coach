from flask import Blueprint, request, jsonify, current_app
from models.utils import predict_calories, calculate_bmi, calculate_nutrition, suggest_workout, decode_jwt
from datetime import datetime
import logging
from bson import ObjectId

prediction_bp = Blueprint('prediction', __name__)
logger = logging.getLogger(__name__)

@prediction_bp.route('', methods=['POST'])
def predict():
    data = request.json
    required = ['age', 'steps', 'weight', 'height', 'activity_level', 'gender', 'heart_rate', 'sleep_hours', 'stress_level', 'mood']
    if not all(k in data for k in required):
        return jsonify({"error": f"Missing required fields: {', '.join(set(required) - set(data.keys()))}"}), 400
    
    for field in ['age', 'steps', 'weight', 'height', 'heart_rate', 'sleep_hours', 'stress_level']:
        if not isinstance(data[field], (int, float)) or data[field] < 0:
            return jsonify({"error": f"Invalid {field}: must be a non-negative number"}), 400
    
    activity_level_map = {'Sedentary': 0, 'Lightly Active': 1, 'Active': 2, 'Very Active': 3}
    gender_map = {'M': 0, 'F': 1}
    mood_map = {'happy': 0, 'neutral': 1, 'sad': 2, 'anxious': 3}
    if (data['activity_level'] not in activity_level_map or 
        data['gender'] not in gender_map or 
        data['mood'] not in mood_map):
        return jsonify({"error": "Invalid activity level, gender, or mood"}), 400
    
    data['activity_level_encoded'] = current_app.label_encoder_activity.transform([data['activity_level']])[0]
    data['gender_encoded'] = current_app.label_encoder_gender.transform([data['gender']])[0]
    data['mood_encoded'] = current_app.label_encoder_mood.transform([data['mood']])[0]
    
    calorie_intake_goal = predict_calories(current_app, data)
    bmi = calculate_bmi(data['weight'], data['height'])
    nutrition = calculate_nutrition(calorie_intake_goal, data['activity_level'])
    
    state = {
        'activity_level': data['activity_level_encoded'],
        'stress_level': data['stress_level'],
        'sleep_hours': data['sleep_hours'],
        'mood': data['mood_encoded']
    }
    workout_suggestion, workout_action = suggest_workout(current_app, state)
    
    token = request.headers.get('Authorization', '').split()
    if len(token) != 2 or token[0].lower() != 'bearer':
        return jsonify({"error": "Token missing"}), 401
    
    decoded_token = decode_jwt(current_app, token[1])
    if not decoded_token:
        return jsonify({"error": "Invalid or expired token"}), 401
    
    user_id = ObjectId(decoded_token['user_id'])
    current_app.users_collection.update_one(
        {"_id": user_id},
        {"$set": {
            "bmi": bmi,
            "height": data['height'],
            "weight": data['weight'],
            "calorie_goal": calorie_intake_goal,
            "nutrition": nutrition,
            "workout_suggestion": workout_suggestion,
            "last_prediction_date": datetime.utcnow(),
            "age": data['age'],
            "activity_level": data['activity_level'],
            "steps": data['steps'],
            "gender": data['gender'],
            "heart_rate": data['heart_rate'],
            "sleep_hours": data['sleep_hours'],
            "stress_level": data['stress_level'],
            "mood": data['mood'],
            "rl_action": workout_action
        }}
    )
    logger.info(f"Prediction made for user: {user_id}")
    return jsonify({
        "calorie_intake_goal": calorie_intake_goal,
        "bmi": bmi,
        "nutrition": nutrition,
        "workout_suggestion": workout_suggestion
    }), 200