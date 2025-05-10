from flask import Flask, render_template
from flask_cors import CORS
from flask_mail import Mail
from pymongo import MongoClient
import tensorflow as tf
import joblib
import os
from dotenv import load_dotenv
from models.r1_agent import RLAgent

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

    # Flask-Mail Configuration
    app.config['MAIL_SERVER'] = os.getenv("MAIL_SERVER", "smtp.gmail.com")
    app.config['MAIL_PORT'] = int(os.getenv("MAIL_PORT", 465))
    app.config['MAIL_USE_TLS'] = os.getenv("MAIL_USE_TLS", "True").lower() == "true"
    app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
    app.config['MAIL_PASSWORD'] = os.getenv("MAIL_PASSWORD")
    app.mail = Mail(app)

    # MongoDB Configuration
    client = MongoClient(os.getenv("MONGO_URL", "mongodb://localhost:27017/"))
    app.db = client['wellness_db']
    app.users_collection = app.db['users']
    app.logs_collection = app.db['logs']
    app.forum_posts_collection = app.db['forum_posts']

    # Load TensorFlow Model and Preprocessors
    app.model = tf.keras.models.load_model('calorie_predictor_v9.keras')
    app.scaler = joblib.load('scaler.joblib')
    app.label_encoder_activity = joblib.load('label_encoder_activity.joblib')
    app.label_encoder_gender = joblib.load('label_encoder_gender.joblib')
    app.label_encoder_mood = joblib.load('label_encoder_mood.joblib')

    # Initialize RL Agent
    app.rl_agent = RLAgent(actions=['yoga', 'cardio', 'strength', 'rest'], 
                           learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1)

    # Register Blueprints
    from auth.routes import auth_bp
    from prediction.routes import prediction_bp
    from profile.routes import profile_bp
    from wellness.routes import wellness_bp
    from admin.routes import admin_bp
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(prediction_bp, url_prefix='/api/predict')
    app.register_blueprint(profile_bp, url_prefix='/api/profile')
    app.register_blueprint(wellness_bp, url_prefix='/api/wellness')
    app.register_blueprint(admin_bp, url_prefix='/api/admin')

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.teardown_appcontext
    def shutdown_session(exception=None):
        app.rl_agent.save_q_table()

    return app