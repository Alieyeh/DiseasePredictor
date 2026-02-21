from flask import Flask
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.debug = True

from blog.views import views
from blog.models import models
from blog.forms import forms

from blog.views.index import index_page

app.register_blueprint(index_page)



