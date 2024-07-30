from flask import Flask, render_template, request, redirect, flash, url_for, render_template_string
import urllib.request
from werkzeug.utils import secure_filename
from main_detect_fish import obj_model_predict
from main_classify_fish import getPrediction
import os
import numpy as np
import cv2
import wikipedia

fish_species_app = Flask(__name__)

@fish_species_app.route("/")
@fish_species_app.route("/home")
def home():
    return render_template("index.html")


@fish_species_app.route('/predict_fish_form',  methods=['GET'])
def predict_fish_form():
    return render_template('case-single.html', predicted=False)

@fish_species_app.route('/count_fish_form',  methods=['GET'])
def count_fish_form():
    return render_template('showcases.html', predicted=False)


@fish_species_app.route('/classify_fish', methods=['POST'])
def classify_fish():
    print(request.url)
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.referrer)
        file = request.files['image']
        if file:
            # convert string of image data to uint8
            nparr = np.fromstring(request.files['image'].read(), np.uint8)
            # decode image
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            label1, acc1 = getPrediction(img)
            flash(label1)
            flash(acc1)
            summary = wikipedia.summary(label1)
            return render_template("blog-single.html", label1=label1, acc1=acc1, summary=summary)


@fish_species_app.route('/count_fish', methods=['POST'])
def count_fish():
    print(request.url)
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.referrer)
        file = request.files['image']
        if file:
            # convert string of image data to uint8
            nparr = np.fromstring(request.files['image'].read(), np.uint8)
            # decode image
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            cv2.imwrite('fish.jpg', img)
            filename, count = obj_model_predict('fish.jpg')
            print(filename)
            print(count)
            return render_template("blog-double.html", filename=filename, count=count)


if __name__ == "__main__":
    fish_species_app.secret_key = 'super secret key'
    fish_species_app.run(debug=True)