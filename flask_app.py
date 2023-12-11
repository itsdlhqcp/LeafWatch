import base64
import datetime
from io import BytesIO
from flask import Flask, render_template
from flask import request
import os
import numpy as np
import pandas as pd
import scipy
import sklearn
import os
from PIL import Image
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
UPLOAD_FOLDER = os.path.join(BASE_PATH, 'static/image/')

##------------------------------LOAD MODELS -----------------------------
model_sgd_path = os.path.join(MODEL_PATH,'dsa_image_classification_sgd.pickle')
scaler_path = os.path.join(MODEL_PATH,'dsa_scaler.pickle')
model_sgd = pickle.load(open(model_sgd_path,'rb'))
scaler = pickle.load(open(scaler_path,'rb'))

# uploaded Images folder path
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'image')
# Check if the folder directory exists, if not then create it
if not os.path.exists(app.config['UPLOAD_FOLDER'] ):
    os.makedirs(app.config['UPLOAD_FOLDER'] )


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

# @app.route('/')
# def about():
#     return render_template("capture.html")

@app.route('/capture', methods=['GET', 'POST'])
def capture():
    filename = ''  # using filename variable to display video feed and captured image alternatively on the same page
    image_data_url = request.form.get('image')

    if request.method == 'POST' and image_data_url:
        try:
            # Decode the base64 data URL to obtain the image data
            image_data = base64.b64decode(image_data_url.split(',')[1])
            
            # Create an image from the decoded data
            img = Image.open(BytesIO(image_data))
            
            # Convert the image to RGB mode if it's in RGBA mode
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            
            # Generate a filename with the current date and time
            timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            upload_file_name = f"leaf_{timestamp}.jpg"
            
            # Save the image to the upload folder
            upload_file_path = os.path.join(UPLOAD_PATH, upload_file_name)
            img.save(upload_file_path, 'JPEG')
            print('File saved successfully')

            # Send to the pipeline model
            results = pipeline_model(upload_file_path, scaler, model_sgd)
            hei = getheight(upload_file_path)
            print(results)

            # Display the results on the template
            return render_template('capture.html', fileupload=True, extension=False, data=results,image_filename=upload_file_name,height=hei)
            
        except IndexError as e:
            error_message = f'Error processing image: {str(e)}'
            return render_template('capture.html', filename=filename, error_message=error_message)

    return render_template('capture.html', filename=filename)
    

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