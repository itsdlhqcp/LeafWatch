from flask import Flask, render_template
from flask import request
import os
app = Flask(__name__)

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH,'static/upload/')

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


        return render_template('upload.html')
    else:
        return render_template('upload.html')

      
    
        

if __name__ == "__main__":
    app.run(debug=True)