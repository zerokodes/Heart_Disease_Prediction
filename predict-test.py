#!/usr/bin/env python
# coding: utf-8

import requests

url = 'http://localhost:9696/predict'

patient = {
    'age': 70,
    'sex': 'male',
    'chest_pain_type': 4,
    'bp': 130,
    'cholesterol': 322,
    'fbs_over_120': 'false',
    'ekg_results': 2,
    'max_hr': 109,
    'exercise_angina': 'n0',
    'st_depression': 2.4,
    'slope_of_st': 2,
    'number_of_vessels_fluro': 3,
    'thallium': 3
}

response = requests.post(url, json=patient).json()
print(response)