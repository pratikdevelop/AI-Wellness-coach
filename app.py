import os
import logging
import json
import requests
from datetime import datetime, timedelta
from flask import Flask, render_template, Blueprint, request, jsonify, current_app
from flask_cors import CORS
from flask_mail import Mail, Message
from flask_jwt_extended import JWTManager, create_access_token, create_refresh_token, jwt_required, get_jwt_identity
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import tensorflow as tf
import joblib
import numpy as np
from dotenv import load_dotenv
import bcrypt
import jwt
import re
from llama_cpp import Llama
from collections import defaultdict

# Suppress TensorFlow oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Utility Functions (from app/models/utils.py) ---
def decode_jwt(token):
    try:
        decoded = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=['HS256'])
        return decoded
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def get_fitbit_data(access_token, user_id):
    headers = {'Authorization': f'Bearer {access_token}'}
    date = datetime.utcnow().strftime('%Y-%m-%d')
    try:
        response = requests.get(
            f'https://api.fitbit.com/1/user/{user_id}/activities/date/{date}.json',
            headers=headers
        )
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Fitbit API error: {response.status_code}")
            return None
    except requests.RequestException as e:
        logger.error(f"Fitbit request failed: {str(e)}")
        return None

# --- RL Agent (from app/models/r1_agent.py) ---
class RLAgent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.q_table = defaultdict(lambda: np.zeros(len(actions)))
        self.q_table_path = 'q_table.pkl'
        self.load_q_table()

    def discretize_state(self, state):
        activity_level = min(int(state['activity_level'] / 3), 3)
        stress_level = min(int(state['stress_level'] / 3), 3)
        sleep_hours = min(int(state['sleep_hours'] / 2), 4)
        mood_idx = state['mood_idx']
        return (activity_level, stress_level, sleep_hours, mood_idx)

    def get_action(self, state):
        state_tuple = self.discretize_state(state)
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        q_values = self.q_table[state_tuple]
        return self.actions[np.argmax(q_values)]

    def update_q_table(self, state, action, reward, next_state):
        state_tuple = self.discretize_state(state)
        next_state_tuple = self.discretize_state(next_state)
        action_idx = self.actions.index(action)
        current_q = self.q_table[state_tuple][action_idx]
        next_max_q = np.max(self.q_table[next_state_tuple])
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state_tuple][action_idx] = new_q

    def save_q_table(self):
        try:
            with open(self.q_table_path, 'wb') as f:
                joblib.dump(dict(self.q_table), f)
            logger.info("Q-table saved successfully")
        except Exception as e:
            logger.error(f"Failed to save Q-table: {str(e)}")

    def load_q_table(self):
        if os.path.exists(self.q_table_path):
            try:
                with open(self.q_table_path, 'rb') as f:
                    self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), joblib.load(f))
                logger.info("Q-table loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Q-table: {str(e)}")

# --- Flask App Initialization (from app/__init__.py) ---
def create_app():
    app = Flask(__name__)
    CORS(app)
    load_dotenv()

    # App Configuration
    app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", "your-secret-key")
    app.config['UPLOAD_FOLDER'] = 'app/static/profile_pics'
    app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    app.config['JWT_EXPIRATION_DELTA'] = int(os.getenv("JWT_EXPIRATION_DELTA", 86400))
    app.config['SPOONACULAR_API_KEY'] = os.getenv("SPOONACULAR_API_KEY")
    app.config['PALMYRA_MED_ENDPOINT'] = os.getenv("PALMYRA_MED_ENDPOINT", "http://localhost:8000/infer")

    # Flask-Mail Configuration
    app.config['MAIL_SERVER'] = os.getenv("MAIL_SERVER", "smtp.gmail.com")
    app.config['MAIL_PORT'] = int(os.getenv("MAIL_PORT", 465))
    app.config['MAIL_USE_TLS'] = os.getenv("MAIL_USE_TLS", "True").lower() == "true"
    app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
    app.config['MAIL_PASSWORD'] = os.getenv("MAIL_PASSWORD")
    app.mail = Mail(app)

    # JWT Configuration
    app.config['JWT_SECRET_KEY'] = app.config['SECRET_KEY']
    jwt = JWTManager(app)

    # MongoDB Configuration
    try:
        client = MongoClient(os.getenv("MONGO_URL", "mongodb://localhost:27017/"), serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        app.db = client['wellness_db']
        app.users_collection = app.db['users']
        app.logs_collection = app.db['logs']
        app.forum_posts_collection = app.db['forum_posts']
        logger.info("MongoDB connection established")
    except ConnectionFailure as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
        raise

    # Load TensorFlow Model and Preprocessors
    try:
        app.model = tf.keras.models.load_model('calorie_predictor_v9.keras')
        app.scaler = joblib.load('scaler.joblib')
        app.label_encoder_activity = joblib.load('label_encoder_activity.joblib')
        app.label_encoder_gender = joblib.load('label_encoder_gender.joblib')
        app.label_encoder_mood = joblib.load('label_encoder_mood.joblib')
        logger.info("TensorFlow model and preprocessors loaded")
    except Exception as e:
        logger.error(f"Failed to load TensorFlow model or preprocessors: {str(e)}")
        raise

    # Initialize RL Agent
    try:
        app.rl_agent = RLAgent(
            actions=['yoga', 'cardio', 'strength', 'rest'],
            learning_rate=0.1,
            discount_factor=0.9,
            exploration_rate=0.1
        )
        logger.info("RL agent initialized")
    except Exception as e:
        logger.error(f"Failed to initialize RL agent: {str(e)}")
        raise

    # Load Mistral-7B GGUF Model
    try:
        app.llm = Llama(
            model_path="i1-Q4_K_M.gguf",
            n_ctx=4096,
            n_threads=8,
            n_gpu_layers=35
        )
        logger.info("Mistral-7B GGUF model loaded")
    except Exception as e:
        logger.error(f"Error loading Mistral GGUF model: {str(e)}. Text-based routes may be limited.")
        app.llm = None

    # --- Authentication Routes (from app/auth/routes.py) ---
    auth_bp = Blueprint('auth', __name__)

    @auth_bp.route('/signup', methods=['POST'])
    def signup():
        try:
            data = request.form
            name = data.get('name')
            username = data.get('username')
            email = data.get('email')
            phone = data.get('phone')
            password = data.get('password')
            confirm_password = data.get('confirmPassword')
            profile_pic = request.files.get('profilePic')

            if not all([name, username, email, phone, password, confirm_password]):
                return jsonify({"error": "All fields are required"}), 400
            if password != confirm_password:
                return jsonify({"error": "Passwords do not match"}), 400

            if current_app.users_collection.find_one({"$or": [{"email": email}, {"username": username}]}):
                return jsonify({"error": "User already exists"}), 400

            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            user_data = {
                "name": name,
                "username": username,
                "email": email,
                "phone": phone,
                "password": hashed_password,
                "mood": "neutral",
                "recent_moods": [],
                "created_at": datetime.utcnow(),
                "profile_pic": ""
            }

            if profile_pic and '.' in profile_pic.filename and profile_pic.filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']:
                filename = f"{username}_{profile_pic.filename}"
                profile_pic.save(os.path.join(current_app.config['UPLOAD_FOLDER'], filename))
                user_data['profile_pic'] = filename

            user_id = current_app.users_collection.insert_one(user_data).inserted_id
            access_token = create_access_token(identity=str(user_id), expires_delta=timedelta(seconds=current_app.config['JWT_EXPIRATION_DELTA']))
            refresh_token = create_refresh_token(identity=str(user_id))

            msg = Message('Welcome to Wellness App!', sender=current_app.config['MAIL_USERNAME'], recipients=[email])
            msg.body = f'Hi {name},\n\nThank you for signing up! Your account has been created successfully.'
            try:
                current_app.mail.send(msg)
                logger.info(f"Welcome email sent to {email}")
            except Exception as e:
                logger.error(f"Failed to send welcome email: {str(e)}")

            logger.info(f"User {username} signed up successfully")
            return jsonify({
                "message": "User created successfully",
                "access_token": access_token,
                "refresh_token": refresh_token,
                "user": {
                    "name": name,
                    "username": username,
                    "email": email,
                    "phone": phone,
                    "mood": "neutral"
                }
            }), 201
        except Exception as e:
            logger.error(f"Signup error: {str(e)}")
            return jsonify({"error": "An error occurred during signup"}), 500

    @auth_bp.route('/login', methods=['POST'])
    def login():
        try:
            data = request.get_json()
            identifier = data.get('identifier')
            password = data.get('password')

            if not identifier or not password:
                return jsonify({"error": "Username/email and password are required"}), 400

            user = current_app.users_collection.find_one({
                "$or": [{"email": identifier}, {"username": identifier}]
            })

            if not user or not bcrypt.checkpw(password.encode('utf-8'), user['password']):
                return jsonify({"error": "Invalid credentials"}), 401

            user_id = str(user['_id'])
            access_token = create_access_token(identity=user_id, expires_delta=timedelta(seconds=current_app.config['JWT_EXPIRATION_DELTA']))
            refresh_token = create_refresh_token(identity=user_id)

            logger.info(f"User {user['username']} logged in successfully")
            return jsonify({
                "message": "Login successful",
                "access_token": access_token,
                "refresh_token": refresh_token,
                "user": {
                    "name": user['name'],
                    "username": user['username'],
                    "email": user['email'],
                    "phone": user['phone'],
                    "mood": user.get('mood', 'neutral')
                }
            }), 200
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            return jsonify({"error": "An error occurred during login"}), 500

    @auth_bp.route('/refresh', methods=['POST'])
    @jwt_required(refresh=True)
    def refresh():
        try:
            current_user_id = get_jwt_identity()
            new_access_token = create_access_token(identity=current_user_id, expires_delta=timedelta(seconds=current_app.config['JWT_EXPIRATION_DELTA']))
            logger.info(f"Token refreshed for user ID {current_user_id}")
            return jsonify({"access_token": new_access_token}), 200
        except Exception as e:
            logger.error(f"Refresh error: {str(e)}")
            return jsonify({"error": "An error occurred during token refresh"}), 500

    # --- Prediction Routes (from app/prediction/routes.py) ---
    prediction_bp = Blueprint('prediction', __name__)

    @prediction_bp.route('/predict', methods=['POST'])
    @jwt_required()
    def predict_calories():
        try:
            user_id = get_jwt_identity()
            data = request.get_json()

            required_fields = ['age', 'steps', 'weight', 'height', 'activity_level', 'gender', 'heart_rate', 'sleep_hours', 'stress_level', 'mood']
            if not all(field in data for field in required_fields):
                return jsonify({"error": "All fields are required"}), 400

            try:
                age = int(data['age'])
                steps = int(data['steps'])
                weight = float(data['weight'])
                height = float(data['height'])
                heart_rate = int(data['heart_rate'])
                sleep_hours = float(data['sleep_hours'])
                stress_level = int(data['stress_level'])
                if not (1 <= stress_level <= 10):
                    return jsonify({"error": "Stress level must be between 1 and 10"}), 400
            except (ValueError, TypeError):
                return jsonify({"error": "Invalid numerical input"}), 400

            activity_level = data['activity_level']
            gender = data['gender']
            mood = data['mood']

            if activity_level not in current_app.label_encoder_activity.classes_:
                return jsonify({"error": "Invalid activity level"}), 400
            if gender not in current_app.label_encoder_gender.classes_:
                return jsonify({"error": "Invalid gender"}), 400
            if mood not in current_app.label_encoder_mood.classes_:
                return jsonify({"error": "Invalid mood"}), 400

            activity_level_encoded = current_app.label_encoder_activity.transform([activity_level])[0]
            gender_encoded = current_app.label_encoder_gender.transform([gender])[0]
            mood_encoded = current_app.label_encoder_mood.transform([mood])[0]

            input_data = np.array([[age, steps, weight, height, activity_level_encoded, gender_encoded, heart_rate, sleep_hours, stress_level, mood_encoded]])
            input_scaled = current_app.scaler.transform(input_data)
            prediction = current_app.model.predict(input_scaled, verbose=0)[0][0]

            current_app.users_collection.update_one(
                {"_id": user_id},
                {
                    "$set": {
                        "age": age,
                        "weight": weight,
                        "height": height,
                        "gender": gender,
                        "mood": mood,
                        "recent_moods": current_app.users_collection.find_one({"_id": user_id}).get('recent_moods', [])[-4:] + [mood]
                    }
                }
            )

            current_app.logs_collection.insert_one({
                "user_id": user_id,
                "prediction": float(prediction),
                "inputs": data,
                "date": datetime.utcnow()
            })

            logger.info(f"Calorie prediction made for user ID {user_id}: {prediction}")
            return jsonify({"calorie_goal": round(float(prediction), 2)}), 200
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return jsonify({"error": "An error occurred during prediction"}), 500

    @prediction_bp.route('/suggest_workout', methods=['POST'])
    @jwt_required()
    def suggest_workout():
        try:
            user_id = get_jwt_identity()
            data = request.get_json()

            required_fields = ['activity_level', 'stress_level', 'sleep_hours', 'mood']
            if not all(field in data for field in required_fields):
                return jsonify({"error": "All fields are required"}), 400

            try:
                activity_level = float(data['activity_level'])
                stress_level = float(data['stress_level'])
                sleep_hours = float(data['sleep_hours'])
                if not (1 <= stress_level <= 10):
                    return jsonify({"error": "Stress level must be between 1 and 10"}), 400
            except (ValueError, TypeError):
                return jsonify({"error": "Invalid numerical input"}), 400

            mood = data['mood']
            if mood not in current_app.label_encoder_mood.classes_:
                return jsonify({"error": "Invalid mood"}), 400
            mood_idx = current_app.label_encoder_mood.transform([mood])[0]

            state = {
                'activity_level': activity_level,
                'stress_level': stress_level,
                'sleep_hours': sleep_hours,
                'mood_idx': mood_idx
            }

            workout = current_app.rl_agent.get_action(state)
            reward = 1.0  # Placeholder reward
            next_state = state  # Simplified
            current_app.rl_agent.update_q_table(state, workout, reward, next_state)

            logger.info(f"Workout suggested for user ID {user_id}: {workout}")
            return jsonify({"suggested_workout": workout}), 200
        except Exception as e:
            logger.error(f"Workout suggestion error: {str(e)}")
            return jsonify({"error": "An error occurred during workout suggestion"}), 500

    @prediction_bp.route('/predict_text', methods=['POST'])
    @jwt_required()
    def predict_text():
        try:
            user_id = get_jwt_identity()
            data = request.get_json()
            text = data.get('symptoms', '')

            if not text:
                return jsonify({"error": "Symptoms description required"}), 400

            if not current_app.llm:
                return jsonify({"error": "Text processing model unavailable"}), 503

            prompt = f"<s>[INST] Based on the symptom description, classify mood as 'happy', 'neutral', 'sad', or 'anxious', and estimate stress level (1-10). Description: {text} [/INST]"
            output = current_app.llm(prompt, max_tokens=100, stop=["</s>"])['choices'][0]['text']
            mood_match = re.search(r"mood:\s*(happy|neutral|sad|anxious)", output, re.IGNORECASE)
            stress_match = re.search(r"stress level:\s*(\d+)", output)

            if not (mood_match and stress_match):
                return jsonify({"error": "Failed to parse model output"}), 500

            mood = mood_match.group(1).lower()
            stress_level = int(stress_match.group(1))

            if mood not in current_app.label_encoder_mood.classes_:
                return jsonify({"error": "Invalid mood detected"}), 400
            if not (1 <= stress_level <= 10):
                return jsonify({"error": "Stress level must be between 1 and 10"}), 400

            current_app.logs_collection.insert_one({
                "user_id": user_id,
                "symptoms": text,
                "mood": mood,
                "stress_level": stress_level,
                "date": datetime.utcnow()
            })

            logger.info(f"Text-based prediction for user ID {user_id}: mood={mood}, stress_level={stress_level}")
            return jsonify({
                "mood": mood,
                "stress_level": stress_level,
                "disclaimer": "This is not medical advice; consult a healthcare provider."
            }), 200
        except Exception as e:
            logger.error(f"Text prediction error: {str(e)}")
            return jsonify({"error": "An error occurred during text prediction"}), 500

    # --- Profile Routes (from app/profile/routes.py) ---
    profile_bp = Blueprint('profile', __name__)

    @profile_bp.route('/profile', methods=['GET'])
    @jwt_required()
    def get_user_profile():
        try:
            user_id = get_jwt_identity()
            user = current_app.users_collection.find_one({"_id": user_id})

            if not user:
                return jsonify({"error": "User not found"}), 404

            user_profile = {
                "name": user.get('name'),
                "username": user.get('username'),
                "email": user.get('email'),
                "phone": user.get('phone'),
                "profile_pic": user.get('profile_pic', ''),
                "mood": user.get('mood', 'neutral'),
                "recent_moods": user.get('recent_moods', []),
                "age": user.get('age'),
                "weight": user.get('weight'),
                "height": user.get('height'),
                "gender": user.get('gender'),
                "bmi": round(user['weight'] / ((user['height'] / 100) ** 2), 2) if user.get('weight') and user.get('height') else None,
                "steps": user.get('steps', 0)
            }

            profile_text = f"User {user['username']} is {user.get('age', 'unknown')} years old, with a BMI of {user_profile['bmi'] or 0}. Recent mood: {user.get('mood', 'neutral')}. Steps: {user.get('steps', 0)}."
            summary = "Summary unavailable"
            if current_app.config['PALMYRA_MED_ENDPOINT']:
                try:
                    response = requests.post(
                        current_app.config['PALMYRA_MED_ENDPOINT'],
                        json={
                            "text": f"Summarize this user profile: {profile_text}",
                            "max_tokens": 100,
                            "temperature": 0.7
                        }
                    )
                    if response.status_code == 200:
                        summary = response.json().get('output', 'Summary unavailable')
                except Exception as e:
                    logger.error(f"Palmyra summary error: {str(e)}")

            user_profile['summary'] = summary
            logger.info(f"Profile retrieved for user ID {user_id}")
            return jsonify(user_profile), 200
        except Exception as e:
            logger.error(f"Profile retrieval error: {str(e)}")
            return jsonify({"error": "An error occurred while retrieving profile"}), 500

    # --- Wellness Routes (from app/wellness/routes.py) ---
    wellness_bp = Blueprint('wellness', __name__)

    @wellness_bp.route('/ask', methods=['POST'])
    def ask_medical_question():
        try:
            data = request.get_json()
            if 'question' not in data:
                return jsonify({"error": "Question required"}), 400

            response = requests.post(
                current_app.config['PALMYRA_MED_ENDPOINT'],
                json={
                    "text": data['question'],
                    "max_tokens": 200,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            )
            if response.status_code != 200:
                return jsonify({"error": "Failed to get response from model"}), 500

            answer = response.json().get('output', '')
            current_app.logs_collection.insert_one({
                "user_id": None,
                "question": data['question'],
                "answer": answer,
                "date": datetime.utcnow()
            })
            logger.info(f"Medical question answered: {data['question']}")
            return jsonify({
                "answer": answer,
                "disclaimer": "This is not medical advice; consult a healthcare provider."
            }), 200
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return jsonify({"error": "An error occurred"}), 500

    # --- Admin Routes (from app/admin/routes.py) ---
    admin_bp = Blueprint('admin', __name__)

    @admin_bp.route('', methods=['GET'])
    def admin_dashboard():
        return jsonify({"message": "Admin endpoint"}), 200

    # Register Blueprints
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(prediction_bp, url_prefix='/api/predict')
    app.register_blueprint(profile_bp, url_prefix='/api/profile')
    app.register_blueprint(wellness_bp, url_prefix='/api/wellness')
    app.register_blueprint(admin_bp, url_prefix='/api/admin')

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/api/training_plot', methods=['GET'])
    def serve_training_plot():
        try:
            return send_file('training_data_plot.png', mimetype='image/png')
        except FileNotFoundError:
            return jsonify({"error": "Training plot not found"}), 404

    @app.teardown_appcontext
    def shutdown_session(exception=None):
        try:
            app.rl_agent.save_q_table()
            logger.info("RL Q-table saved")
        except Exception as e:
            logger.error(f"Failed to save RL Q-table: {str(e)}")

    logger.info("Flask app initialized successfully")
    return app

# Run the app
if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)