from flask import Blueprint

prediction_bp = Blueprint('prediction', __name__)

from . import routes