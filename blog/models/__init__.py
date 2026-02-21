from flask import Blueprint

models = Blueprint('models', __name__)

from blog.models import models
