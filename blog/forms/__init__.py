from flask import Blueprint

forms = Blueprint('froms', __name__)

from blog.forms import forms
