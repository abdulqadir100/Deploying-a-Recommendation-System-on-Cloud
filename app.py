#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from flask import Flask,render_template,request
from recommendation_engine import Movie_recomendation_system#.recommendation_by_favourite_movie as movie_recommender

app = Flask(__name__)
#model = pickle.load(open('lr_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict_expenses',methods=['POST'])
def predict_expenses():
    features = [x for x in request.form.values()]
    target_user_id = int(features[0])
    #movie_title =  features[1]
    output = Movie_recomendation_system.automatic_recommendation(target_user_id)

    return render_template('index.html', prediction_text='Recommended movies {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)

