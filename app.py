from flask import Flask , render_template , request,url_for
import os
app = Flask(__name__)
app.config['UPLOAD_FOLDER']="static/images"
@app.route('/')
def index():
    return render_template("index.html")
@app.route('/predict' ,methods=["POST"])
def predict():
    file=request.files['img']
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    pth_img=os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    uploaded=True
    return render_template("index.html",pth_img=pth_img,uploaded=uploaded)

if __name__=="__main__":
    app.run(debug=False)