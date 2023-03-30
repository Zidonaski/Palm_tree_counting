from flask import Flask , render_template , request,url_for
import os
from model import CNN
from utils import *
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
    prediction=str( round( Treecounting_model(img_preprocessed).item() ) )
    predict=True
    return render_template("index.html",pth_img=pth_img,uploaded=uploaded,predict=predict,prediction=prediction)

if __name__=="__main__":
    Treecounting_model=CNN()
    Treecounting_model.load_state_dict(torch.load("static/models/best_model.pt",map_location=torch.device('cpu')))
    Treecounting_model.eval()
    app.run(debug=False)