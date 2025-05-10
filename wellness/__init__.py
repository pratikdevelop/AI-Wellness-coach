from flask import Blueprint

wellness_bp = Blueprint('wellness', __name__)

from . import routes