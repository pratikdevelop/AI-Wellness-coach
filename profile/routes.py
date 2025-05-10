from flask import Blueprint, request, jsonify, current_app
from models.utils import get_fitbit_data, decode_jwt
import statistics
from bson import ObjectId
import logging

profile_bp = Blueprint('profile', __name__)
logger = logging.getLogger(__name__)

@profile_bp.route('', methods=['GET'])
def get_user_profile():
    try:
        # Validate Authorization header
        token = request.headers.get('Authorization', '').split()
        if len(token) != 2 or token[0].lower() != 'bearer':
            return jsonify({"error": "Token missing"}), 401
        
        # Decode JWT
        decoded_token = decode_jwt(current_app, token[1])
        if not decoded_token:
            return jsonify({"error": "Invalid or expired token"}), 401
        
        # Fetch user
        user_id = ObjectId(decoded_token['user_id'])
        user = current_app.users_collection.find_one({"_id": user_id})
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        # Calculate progress for goals
        goals = user.get('goals', {})
        progress = {}
        if 'weight' in goals:
            progress['weight'] = {"goal": goals['weight'], "current": user.get('weight', 0)}
        if 'steps' in goals:
            progress['steps'] = {"goal": goals['steps'], "current": user.get('steps', 0)}
        if 'water_intake' in goals:
            progress['water_intake'] = {"goal": goals['water_intake'], "current": user.get('water_intake', 0)}
        
        # Fetch recent moods and average stress
        recent_moods = [
            log['mood'] for log in current_app.logs_collection.find(
                {"user_id": user_id, "mood": {"$exists": True}}
            ).sort("date", -1).limit(5)
        ]
        stress_levels = [
            log['stress_level'] for log in current_app.logs_collection.find(
                {"user_id": user_id, "stress_level": {"$exists": True}}
            )
        ]
        avg_stress = round(statistics.mean(stress_levels), 2) if stress_levels else None
        
        # Construct user profile
        user_profile = {
            "username": user.get('username', ''),
            "email": user.get('email', ''),
            "age": user.get('age', 0),
            "weight": user.get('weight', 0),
            "height": user.get('height', 0),
            "calorie_goal": user.get('calorie_goal', 0),
            "workouts": user.get('workout_suggestion', ''),
            "nutrition": user.get('nutrition', {}),
            "fitbit_data": get_fitbit_data(user.get('fitbit_access_token', '')) if user.get('fitbit_access_token') else {},
            "bmi": user.get('bmi', 0),
            "calories_burned": user.get('calories_burned', 0),
            "activity_level": user.get('activity_level', 'Sedentary'),
            "steps": user.get('steps', 0),
            "gender": user.get('gender', 'M'),
            "heart_rate": user.get('heart_rate', 0),
            "sleep_hours": user.get('sleep_hours', 0),
            "stress_level": user.get('stress_level', 0),
            "mood": user.get('mood', 'neutral'),  # Added mood field
            "water_intake": user.get('water_intake', 0),
            "goals": goals,
            "progress": progress,
            "points": user.get('points', 0),
            "recent_moods": recent_moods,
            "average_stress": avg_stress
        }
        logger.info(f"Profile retrieved for user: {user_id}")
        return jsonify(user_profile), 200
    
    except Exception as e:
        logger.error(f"Error retrieving profile for user_id {user_id}: {str(e)}")
        return jsonify({"error": "An error occurred while retrieving profile"}), 500