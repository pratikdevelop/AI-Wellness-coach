from flask import Blueprint, request, jsonify, current_app
from models.utils import generate_jwt, decode_jwt, generate_password_reset_token, send_reset_email, allowed_file
import bcrypt
from datetime import datetime
import logging
from bson import ObjectId
import os
from werkzeug.utils import secure_filename
import uuid

from . import auth_bp

logger = logging.getLogger(__name__)

@auth_bp.route('/signup', methods=['POST'])
def register_user():
    try:
        if 'profilePic' in request.files:
            file = request.files['profilePic']
            if file and allowed_file(file.filename, current_app.config['ALLOWED_EXTENSIONS']):
                filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4()}_{filename}"
                filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                file.save(filepath)
                profile_pic_url = f"/static/profile_pics/{unique_filename}"
            else:
                return jsonify({"error": "Invalid file type"}), 400
        else:
            profile_pic_url = None

        form_data = request.form
        required_fields = ['name', 'username', 'email', 'phone', 'password', 'confirmPassword']
        if not all(field in form_data for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400

        if form_data['password'] != form_data['confirmPassword']:
            return jsonify({"error": "Passwords do not match"}), 400

        if len(form_data['password']) < 8:
            return jsonify({"error": "Password must be at least 8 characters"}), 400

        if current_app.users_collection.find_one({"$or": [{"email": form_data['email']}, {"username": form_data['username']}]}):
            return jsonify({"error": "Email or username already exists"}), 400

        hashed_password = bcrypt.hashpw(form_data['password'].encode('utf-8'), bcrypt.gensalt())
        user_data = {
            "name": form_data['name'],
            "username": form_data['username'],
            "email": form_data['email'],
            "phone": form_data['phone'],
            "password": hashed_password,
            "profile_pic": profile_pic_url,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "age": 0,
            "weight": 0,
            "height": 0,
            "activity_level": "Sedentary",
            "gender": "M",
            "heart_rate": 0,
            "sleep_hours": 0,
            "stress_level": 0,
            "mood": "neutral",
            "goals": {},
            "points": 0
        }
        result = current_app.users_collection.insert_one(user_data)
        token = generate_jwt(current_app, str(result.inserted_id))
        logger.info(f"New user registered: {form_data['email']}")
        return jsonify({
            "message": "User registered successfully",
            "token": token,
            "user": {
                "id": str(result.inserted_id),
                "name": user_data['name'],
                "email": user_data['email'],
                "profile_pic": user_data['profile_pic']
            }
        }), 201
    except Exception as e:
        logger.error(f"Error in signup: {str(e)}")
        return jsonify({"error": "An error occurred during registration"}), 500

@auth_bp.route('/login', methods=['POST'])
def login_user():
    data = request.json
    if not all(k in data for k in ['email', 'password']):
        return jsonify({"error": "Email and password required"}), 400
    
    user = current_app.users_collection.find_one({"email": data['email']})
    if not user or not bcrypt.checkpw(data['password'].encode('utf-8'), user['password']):
        return jsonify({"error": "Invalid email or password"}), 401
    
    token = generate_jwt(current_app, str(user['_id']))
    logger.info(f"User logged in: {data['email']}")
    return jsonify({
        "message": "Login successful",
        "token": token,
        "user": {
            "id": str(user['_id']),
            "name": user.get('name'),
            "email": user.get('email'),
            "profile_pic": user.get('profile_pic')
        }
    }), 200

@auth_bp.route('/reset_password_request', methods=['POST'])
def reset_password_request():
    data = request.json
    user = current_app.users_collection.find_one({"email": data.get('email')})
    if not user:
        return jsonify({"error": "Email not found"}), 404
    
    token = generate_password_reset_token(current_app, data['email'])
    send_reset_email(current_app, data['email'], token)
    logger.info(f"Password reset requested for: {data['email']}")
    return jsonify({"message": "Password reset email sent"}), 200

@auth_bp.route('/reset_password/<token>', methods=['POST'])
def reset_password(token):
    decoded_token = decode_jwt(current_app, token)
    if not decoded_token:
        return jsonify({"error": "Invalid or expired token"}), 401
    
    data = request.json
    if 'password' not in data or len(data['password']) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400
    
    hashed_password = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt())
    current_app.users_collection.update_one(
        {"email": decoded_token['email']},
        {"$set": {"password": hashed_password}}
    )
    logger.info(f"Password reset for: {decoded_token['email']}")
    return jsonify({"message": "Password reset successfully"}), 200