from roboflow import Roboflow
import string
import random
import json
import os
import shutil
 
rf = Roboflow(api_key="K57tUktyTD0l9DKKeQBE")
workspace_ = rf.workspace()
project = workspace_.project("detect-for-me")
version = project.version(1)
model = version.model

def obj_model_predict(image):
    prediction = model.predict(image) 
    res = 'predictions_' + ''.join(random.choices(string.ascii_uppercase +
                                string.digits, k=7)) + '.jpg'
    prediction.save(output_path=res)
    shutil.move('/workspace/fish_species_detection/'+res, "/workspace/fish_species_detection/static/predictions/"+res)
    count = len(prediction.json()['predictions'])
    return "/static/predictions/"+res, str(count)

