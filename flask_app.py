from flask import Flask, render_template
from flask import request
import os
import numpy as np
import pandas as pd
import scipy
import sklearn
import os
import skimage
import skimage.color
import skimage.transform
import skimage.feature
import skimage.io
import pickle
app = Flask(__name__)

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH,'static/upload/')
MODEL_PATH = os.path.join(BASE_PATH, 'static/models/')

##------------------------------LOAD MODELS -----------------------------
model_sgd_path = os.path.join(MODEL_PATH,'dsa_image_classification_sgd.pickle')
scaler_path = os.path.join(MODEL_PATH,'dsa_scaler.pickle')
model_sgd = pickle.load(open(model_sgd_path,'rb'))
scaler = pickle.load(open(scaler_path,'rb'))


@app.errorhandler(404)
def error404(error):
    message="error404"
    return render_template("error.html",message=message)

@app.errorhandler(405)
def error405(error):
    message="error405"
    return render_template("error.html",message=message)

@app.errorhandler(500)
def error500(error):
    message="error500"
    return render_template("error.html",message=message)

@app.route('/',methods=['GET','POST'])
def index():
    if request.method == "POST":
        upload_file =request.files['image_name']
        filename=upload_file.filename
        print('The filename has been uploaded =',filename)
        #knows the extensions of the file
        ext = filename.split('.')[-1]
        print('The extention of the filename=',ext)
        if ext.lower() in ['png','jpg','jpeg']:
            path_save = os.path.join(UPLOAD_PATH,filename)
            upload_file.save(path_save)
            print('File saved sucessfully')
            # send to pipeline model
            results = pipeline_model(path_save,scaler,model_sgd)
            hei = getheight(path_save)
            print(results)
            return render_template('upload.html',fileupload=True,extension=False,data=results,image_filename=filename,height=hei)
        else:
            print('Use only the extention with .jpg, .png, .jpeg')

            return render_template('upload.html',extension=True,fileupload=False)


    else:
        return render_template('upload.html',fileupload=False)

@app.route('/about/')
def about():
    return render_template("about.html")

def getheight(path):
    img = skimage.io.imread(path)
    h,w,_ =img.shape
    aspect = h/w
    given_width = 100
    height = given_width*aspect
    return height


      
    
def pipeline_model(path,scaler_transform,model_sgd):
    # pipeline model
    image = skimage.io.imread(path)
    # transform image into 80 x 80
    image_resize = skimage.transform.resize(image,(330,420))
    image_scale = 255*image_resize
    image_transform = image_scale.astype(np.uint8)
    # rgb to gray
    gray = skimage.color.rgb2gray(image_transform)
    # hog feature
    feature_vector = skimage.feature.hog(gray,
                                  orientations=10,
                                  pixels_per_cell=(8,8),cells_per_block=(2,2))
    # scaling
    
    scalex = scaler_transform.transform(feature_vector.reshape(1,-1))
    result = model_sgd.predict(scalex)
    # decision function # confidence
    decision_value = model_sgd.decision_function(scalex).flatten()
    labels = model_sgd.classes_
    # probability
    z = scipy.stats.zscore(decision_value)
    prob_value = scipy.special.softmax(z)
    
    # top 5
    top_2_prob_ind = prob_value.argsort()[::-1][:2]
    top_labels = labels[top_2_prob_ind]
    top_prob = prob_value[top_2_prob_ind]
    # put in dictornary
    top_dict = dict()
    for key,val in zip(top_labels,top_prob):
        top_dict.update({key:np.round(val,3)})
    
    return top_dict        

if __name__ == "__main__":
    app.run(debug=True)