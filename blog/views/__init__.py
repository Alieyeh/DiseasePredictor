from flask import Blueprint

views = Blueprint('views', __name__)

from blog.views import views
