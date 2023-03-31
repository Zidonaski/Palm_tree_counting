from flask import Flask , render_template , request,url_for
import os
import onnxruntime as ort
from img_utils import preprocess
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
    img_preprocessed=preprocess(pth_img,480)
    output = ort_sess.run(["output"], {'image': img_preprocessed})
    output=round(output[0].item())
    predict=True
    return render_template("index.html",pth_img=pth_img,uploaded=uploaded,predict=predict,prediction=output)

if __name__=="__main__":
    ort_sess = ort.InferenceSession('./static/models/best_model.onnx')
    app.run(debug=False,port=5000)
    
    