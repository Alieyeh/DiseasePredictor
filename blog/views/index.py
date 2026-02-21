# -*- coding: utf-8 -*-
# import sys
# reload(sys)
# sys.setdefaultencoding('utf8')

from flask import Blueprint, render_template, request, jsonify
from blog import app
from blog.models.DiseasePredictor import DiseasePredictor

index_page = Blueprint('index', __name__, template_folder='templates')

diseasePredictor = DiseasePredictor()
diseasePredictor.calculate_matrix_of_HSDN('blog/static/data/weighted_graph.tsv')


@app.route('/abdal')
def index():
    symptomList = []
    with open(
            'blog/static/data/symptoms.txt') as f:
        data = f.read()
        for row in data.split('\n'):
            symptomList.append(row)
    return render_template('index.html', symptomList=symptomList)


@app.route('/searchDisease', methods=['GET', 'POST'])
def searchDisease():
    if request.method == "POST":
        symptoms = request.json['symptoms']
        diseases = diseasePredictor.get_k_top_related_disease(symptoms, 5)
        series = list(disease[1] for disease in diseases)
        labels = list(disease[0] for disease in diseases)
        series.append(100 - sum(series))
        labels.append('others')
    return jsonify({'series': series, 'labels': labels})


@app.route('/searchSymptom', methods=['GET', 'POST'])
def searchSymptom():
    if request.method == "POST":
        symptoms = request.json['symptoms']
        suggested_symptoms = diseasePredictor.get_k_top_related_symptoms(symptoms, 5)
    return jsonify(suggested_symptoms)
