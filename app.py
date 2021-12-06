import os
from flask import Flask, render_template, request, url_for, redirect
from werkzeug.utils import secure_filename
import covid_detection_model

root_folder = os.path.abspath(os.path.dirname(__file__))
print(root_folder)
UPLOAD_FOLDER = os.path.join(root_folder, "static\\uploads\\")
print(UPLOAD_FOLDER)
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def index():
    return render_template("index.html",ospf = 1)

@app.route("/", methods = ['POST'])
def patient():
    if request.method == "POST":
        print(request)
        name = request.form["name"] #taking data from dictionary
        xray = request.files["xray"]
        print("\n")
        filename = secure_filename(xray.filename)
        xray.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        url = os.path.join('static\\uploads', filename)
        # url = os.path.abspath(url)
        print(url)
        absolute_url =  os.path.abspath(url)
        res_list = covid_detection_model.xray_test(absolute_url)
    return render_template("index.html",ospf = 0,n = name,  xray = url, res = res_list)

if __name__ == "__main__":
    app.run()
