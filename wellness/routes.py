from flask import Blueprint, request, jsonify
from models.utils import get_fitbit_data, decode_jwt
from datetime import datetime
import statistics
import requests
import logging
from bson import ObjectId

wellness_bp = Blueprint('wellness', __name__)
logger = logging.getLogger(__name__)

@wellness_bp.route('/goals', methods=['POST'])
def set_goals():
    from app import create_app
    app = create_app()
    token = request.headers.get('Authorization', '').split()
    if len(token) != 2 or token[0].lower() != 'bearer':
        return jsonify({"error": "Token missing"}), 401
    
    decoded_token = decode_jwt(app, token[1])
    if not decoded_token:
        return jsonify({"error": "Invalid or expired token"}), 401
    
    data = request.json
    valid_goals = ['weight', 'steps', 'water_intake']
    goals = {k: v for k, v in data.items() if k in valid_goals and isinstance(v, (int, float)) and v >= 0}
    if not goals:
        return jsonify({"error": "Invalid or missing goal data"}), 400
    
    user_id = ObjectId(decoded_token['user_id'])
    app.users_collection.update_one(
        {"_id": user_id},
        {"$set": {"goals": goals}}
    )
    logger.info(f"Goals set for user: {user_id}")
    return jsonify({"message": "Goals updated successfully!", "goals": goals}), 200

@wellness_bp.route('/log_mood', methods=['POST'])
def log_mood():
    from app import create_app
    app = create_app()
    data = request.json
    token = request.headers.get('Authorization', '').split()
    if len(token) != 2 or token[0].lower() != 'bearer':
        return jsonify({"error": "Token missing"}), 401
    
    decoded_token = decode_jwt(app, token[1])
    if not decoded_token:
        return jsonify({"error": "Invalid or expired token"}), 401
    
    user_id = ObjectId(decoded_token['user_id'])
    user = app.users_collection.find_one({"_id": user_id})
    log_data = {
        'user_id': user_id,
        'date': datetime.utcnow(),
        'mood': data.get('mood'),
        'stress_level': data.get('stress_level'),
        'notes': data.get('notes', '')
    }
    app.logs_collection.insert_one(log_data)
    app.users_collection.update_one({"_id": user_id}, {"$inc": {"points": 10}})

    # RL Feedback: Update Q-table based on mood improvement
    if user.get('rl_action') and data.get('mood'):
        previous_mood_logs = list(app.logs_collection.find({"user_id": user_id, "mood": {"$exists": True}}).sort("date", -1).limit(2))
        if len(previous_mood_logs) > 1:
            mood_map = {'sad': -1, 'anxious': -0.5, 'neutral': 0, 'happy': 1}
            prev_mood = mood_map.get(previous_mood_logs[1]['mood'], 0)
            curr_mood = mood_map.get(data['mood'], 0)
            reward = curr_mood - prev_mood  # Reward based on mood improvement
            prev_state = {
                'activity_level': app.label_encoder_activity.transform([user['activity_level']])[0],
                'stress_level': user['stress_level'],
                'sleep_hours': user['sleep_hours'],
                'mood': previous_mood_logs[1]['mood']
            }
            curr_state = {
                'activity_level': app.label_encoder_activity.transform([user['activity_level']])[0],
                'stress_level': data.get('stress_level', user['stress_level']),
                'sleep_hours': user['sleep_hours'],
                'mood': data['mood']
            }
            app.rl_agent.update_q_table(prev_state, user['rl_action'], reward, curr_state)
            app.rl_agent.save_q_table()

    logger.info(f"Mood logged for user: {user_id}")
    return jsonify({"message": "Mood logged successfully! Points awarded: 10"}), 201

@wellness_bp.route('/mental_health_insights', methods=['GET'])
def mental_health_insights():
    from app import create_app
    app = create_app()
    token = request.headers.get('Authorization', '').split()
    if len(token) != 2 or token[0].lower() != 'bearer':
        return jsonify({"error": "Token missing"}), 401
    
    decoded_token = decode_jwt(app, token[1])
    if not decoded_token:
        return jsonify({"error": "Invalid or expired token"}), 401
    
    user_id = ObjectId(decoded_token['user_id'])
    user = app.users_collection.find_one({"_id": user_id})
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    fitbit_data = get_fitbit_data(user.get('fitbit_access_token', ''))
    mood_logs = list(app.logs_collection.find({"user_id": user_id, "mood": {"$exists": True}}).sort("date", -1).limit(7))
    
    mood_trend = "No mood data available." if not mood_logs else f"Your mood over the last {len(mood_logs)} days has been mostly {statistics.mode([log['mood'] for log in mood_logs])}."
    sleep_stress = "Insufficient data for analysis."
    if fitbit_data and 'sleep_duration' in fitbit_data and mood_logs:
        sleep_duration = fitbit_data['sleep_duration']
        stress_levels = [log['stress_level'] for log in mood_logs if 'stress_level' in log and log['stress_level'] is not None]
        if stress_levels and sleep_duration < 6 and statistics.mean(stress_levels) > 5:
            sleep_stress = "Your sleep duration is low, and stress levels are high. Consider relaxation exercises."
        elif stress_levels:
            sleep_stress = "Your sleep and stress levels are within a healthy range."
    
    suggestion = "Log your mood to receive suggestions." if not mood_logs else (
        "Try a 5-minute breathing exercise to help manage your mood." if mood_logs[0]['mood'] in ['sad', 'anxious'] else 
        "Keep up the good work! Consider a mindfulness exercise to maintain your positive mood."
    )
    
    insights = {
        'recent_mood_trend': mood_trend,
        'sleep_stress_correlation': sleep_stress,
        'suggestion': suggestion
    }
    logger.info(f"Mental health insights generated for user: {user_id}")
    return jsonify(insights), 200

@wellness_bp.route('/meal_plan', methods=['POST'])
def get_meal_plan():
    from app import create_app
    app = create_app()
    token = request.headers.get('Authorization', '').split()
    if len(token) != 2 or token[0].lower() != 'bearer':
        return jsonify({"error": "Token missing"}), 401
    decoded_token = decode_jwt(app, token[1])
    if not decoded_token:
        return jsonify({"error": "Invalid or expired token"}), 401
    user_id = ObjectId(decoded_token['user_id'])
    user = app.users_collection.find_one({"_id": user_id})
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    data = request.json
    calorie_goal = user.get('calorie_goal', 2000)
    diet = data.get('diet', 'balanced')
    url = f"https://api.spoonacular.com/mealplanner/generate?apiKey={app.config['SPOONACULAR_API_KEY']}&timeFrame=day&targetCalories={calorie_goal}&diet={diet}"
    response = requests.get(url)
    if response.status_code == 200:
        return jsonify({"meal_plan": response.json()}), 200
    return jsonify({"error": "Failed to fetch meal plan"}), 500

@wellness_bp.route('/forum/posts', methods=['POST'])
def create_post():
    from app import create_app
    app = create_app()
    token = request.headers.get('Authorization', '').split()
    if len(token) != 2 or token[0].lower() != 'bearer':
        return jsonify({"error": "Token missing"}), 401
    decoded_token = decode_jwt(app, token[1])
    if not decoded_token:
        return jsonify({"error": "Invalid or expired token"}), 401
    data = request.json
    if not data.get('content'):
        return jsonify({"error": "Content is required"}), 400
    post = {
        'user_id': ObjectId(decoded_token['user_id']),
        'content': data['content'],
        'date': datetime.utcnow(),
        'comments': []
    }
    app.forum_posts_collection.insert_one(post)
    return jsonify({"message": "Post created", "post": post}), 201

@wellness_bp.route('/tips', methods=['GET'])
def get_wellness_tips():
    from app import create_app
    app = create_app()
    token = request.headers.get('Authorization', '').split()
    if len(token) != 2 or token[0].lower() != 'bearer':
        return jsonify({"error": "Token missing"}), 401
    decoded_token = decode_jwt(app, token[1])
    if not decoded_token:
        return jsonify({"error": "Invalid or expired token"}), 401
    user_id = ObjectId(decoded_token['user_id'])
    user = app.users_collection.find_one({"_id": user_id})
    if not user:
        return jsonify({"error": "User not found"}), 404
    mood_logs = list(app.logs_collection.find({"user_id": user_id, "mood": {"$exists": True}}).sort("date", -1).limit(1))
    fitbit_data = get_fitbit_data(user.get('fitbit_access_token', ''))
    tips = []
    if mood_logs and mood_logs[0]['mood'] in ['sad', 'anxious']:
        tips.append("Try a 5-minute breathing exercise to manage your mood.")
    if fitbit_data and fitbit_data.get('sleep_duration', 0) < 6:
        tips.append("Aim for at least 7 hours of sleep tonight for better recovery.")
    return jsonify({"tips": tips}), 200

@wellness_bp.route('/badges', methods=['GET'])
def get_badges():
    from app import create_app
    app = create_app()
    token = request.headers.get('Authorization', '').split()
    if len(token) != 2 or token[0].lower() != 'bearer':
        return jsonify({"error": "Token missing"}), 401
    decoded_token = decode_jwt(app, token[1])
    if not decoded_token:
        return jsonify({"error": "Invalid or expired token"}), 401
    user_id = ObjectId(decoded_token['user_id'])
    user = app.users_collection.find_one({"_id": user_id})
    if not user:
        return jsonify({"error": "User not found"}), 404
    mood_logs_count = app.logs_collection.count_documents({"user_id": user_id, "mood": {"$exists": True}})
    badges = []
    if mood_logs_count >= 7:
        badges.append("Mood Tracker Pro")
    return jsonify({"badges": badges}), 200

@wellness_bp.route('/log_progress', methods=['POST'])
def log_progress():
    from app import create_app
    app = create_app()
    data = request.json
    token = request.headers.get('Authorization', '').split()
    if len(token) != 2 or token[0].lower() != 'bearer':
        return jsonify({"error": "Token missing"}), 401
    decoded_token = decode_jwt(app, token[1])
    if not decoded_token:
        return jsonify({"error": "Invalid or expired token"}), 401
    user_id = ObjectId(decoded_token['user_id'])
    user = app.users_collection.find_one({"_id": user_id})
    
    valid_fields = {
        'steps': (int, float), 'calories_consumed': (int, float), 'carbs_consumed': (int, float),
        'protein_consumed': (int, float), 'fat_consumed': (int, float), 'weight': (int, float),
        'water_intake': (int, float), 'sleep_hours': (int, float), 'mood': str, 'stress_level': (int, float),
        'exercise_duration': (int, float), 'exercise_type': str
    }
    log_data = {'user_id': user_id, 'date': datetime.utcnow()}
    for field, types in valid_fields.items():
        if field in data and isinstance(data[field], types) and (not isinstance(data[field], (int, float)) or data[field] >= 0):
            log_data[field] = data[field]
    
    if len(log_data) <= 2:
        return jsonify({"error": "No valid progress data provided"}), 400
    
    app.logs_collection.insert_one(log_data)
    update_fields = {k: v for k, v in log_data.items() if k in ['weight', 'water_intake', 'sleep_hours', 'stress_level']}
    if update_fields:
        app.users_collection.update_one({"_id": user_id}, {"$set": update_fields})

    # RL Feedback: Update Q-table based on progress (e.g., steps goal)
    if user.get('rl_action') and 'steps' in data and 'steps' in user.get('goals', {}):
        prev_state = {
            'activity_level': app.label_encoder_activity.transform([user['activity_level']])[0],
            'stress_level': user['stress_level'],
            'sleep_hours': user['sleep_hours'],
            'mood': 'neutral'  # Default if no prior mood
        }
        curr_state = {
            'activity_level': app.label_encoder_activity.transform([user['activity_level']])[0],
            'stress_level': data.get('stress_level', user['stress_level']),
            'sleep_hours': data.get('sleep_hours', user['sleep_hours']),
            'mood': data.get('mood', 'neutral')
        }
        steps_goal = user['goals']['steps']
        reward = 1 if data['steps'] >= steps_goal else -1  # Simple reward based on goal achievement
        app.rl_agent.update_q_table(prev_state, user['rl_action'], reward, curr_state)
        app.rl_agent.save_q_table()

    logger.info(f"Progress logged for user: {user_id}")
    return jsonify({"message": "Progress logged successfully!"}), 201