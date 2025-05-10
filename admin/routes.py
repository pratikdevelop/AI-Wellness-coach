from flask import Blueprint, request, jsonify
from models.utils import decode_jwt
import logging
from bson import ObjectId

admin_bp = Blueprint('admin', __name__)
logger = logging.getLogger(__name__)

@admin_bp.route('/dashboard', methods=['GET'])
def admin_dashboard():
    from app import create_app
    app = create_app()
    token = request.headers.get('Authorization', '').split()
    if len(token) != 2 or token[0].lower() != 'bearer':
        return jsonify({"error": "Token missing"}), 401
    
    decoded_token = decode_jwt(app, token[1])
    if not decoded_token or not decoded_token.get('is_admin', False):
        return jsonify({"error": "Admin access required"}), 403
    
    users = list(app.users_collection.find({}, {"username": 1, "email": 1, "last_prediction_date": 1, "points": 1}))
    logs = list(app.logs_collection.find({}, {"user_id": 1, "date": 1, "steps": 1, "calories_consumed": 1, "weight": 1, "mood": 1, "stress_level": 1}))
    logger.info("Admin dashboard accessed")
    return jsonify({"users": users, "logs": logs}), 200

@admin_bp.route('/retrain_model', methods=['POST'])
def retrain_model():
    from app import create_app
    app = create_app()
    token = request.headers.get('Authorization', '').split()
    if len(token) != 2 or token[0].lower() != 'bearer':
        return jsonify({"error": "Token missing"}), 401
    
    decoded_token = decode_jwt(app, token[1])
    if not decoded_token or not decoded_token.get('is_admin', False):
        return jsonify({"error": "Admin access required"}), 403
    
    # Load data from MongoDB
    logs = list(app.logs_collection.find({"steps": {"$exists": True}}))
    if not logs:
        return jsonify({"error": "No data available for retraining"}), 400
    
    data = pd.DataFrame([{
        'age': log.get('age', 30),
        'steps': log['steps'],
        'weight': log.get('weight', 70),
        'height': log.get('height', 170),
        'activity_level': log.get('activity_level', 'Sedentary'),
        'gender': log.get('gender', 'M'),
        'heart_rate': log.get('heart_rate', 70),
        'sleep_hours': log.get('sleep_hours', 7),
        'stress_level': log.get('stress_level', 5),
        'calories': log.get('calorie_goal', calculate_calorie_goal(
            log.get('activity_level', 'Sedentary'), log.get('weight', 70),
            log.get('height', 170), log.get('age', 30), log.get('gender', 'M'),
            log.get('sleep_hours', 7), log.get('stress_level', 5)))
    } for log in logs])

    # Encode and scale data (reuse logic from train_model.py)
    data['activity_level_encoded'] = app.label_encoder_activity.fit_transform(data['activity_level'])
    data['gender_encoded'] = app.label_encoder_gender.fit_transform(data['gender'])
    X = data[['age', 'steps', 'weight', 'height', 'activity_level_encoded', 'gender_encoded', 
              'heart_rate', 'sleep_hours', 'stress_level']].values
    y = data['calories'].values
    X_scaled = app.scaler.fit_transform(X)

    # Train the model
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    history = app.model.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_test, y_test), 
                           callbacks=[EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)], verbose=0)
    app.model.save('calorie_predictor_v9.keras')  # Update v9
    joblib.dump(app.scaler, 'scaler.joblib')
    
    logger.info("Model retrained successfully")
    return jsonify({"message": "Model retrained successfully"}), 200