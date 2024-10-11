import os
import argparse
import cv2
import librosa
import librosa.display
import numpy as np
import tensorflow as tf
import matplotlib as plt
plt.use('Agg')
from model import ClassifierModel
import hyperparameters as hp
from preprocess import Datasets
from flask import Flask, render_template, request
from werkzeug import secure_filename

UPLOAD_FOLDER = '/static/data/'
ALLOWED_EXTENSIONS = set(['wav', 'mp3', 'mp4'])

# # create the folders when setting up your app
os.makedirs(os.path.join( 'static/data/'), exist_ok=True)

app = Flask(__name__)
app.secret_key = '9012uwj1fi3n12i9fj1283fj1208fj1' # no idea what this's for
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('upload.html')
    # full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'graph.png')
    # return render_template('results.html', user_image = full_filename)

def predictor(file_name):
    datasets = Datasets(".")

    model = ClassifierModel()
    model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
    mypath = './weights'

    # getting best weight
    files = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    files = list(filter((lambda x: 'weights' in x), files))
    if len(files) == 0:
        raise ValueError("No weights")

    files.sort(reverse=True, key=lambda x: int(x[17:21]))
    best_weight = files[0]
    
    model.load_weights(os.path.join('./weights', best_weight))
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"],
    )

    # converting to spectrogram

    full_name = os.path.join('./static/data/', file_name)
    x, sr = librosa.load(full_name)
    S = librosa.feature.melspectrogram(x)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB)
    plt.pyplot.tight_layout()
    plt.pyplot.savefig('./static/data/test.png')
    plt.pyplot.close()

    # loading png
    path = os.path.join('./static/data/', 'test.png')
    img = cv2.imread(path)
    if img is None:
        raise ValueError("wrong path or image does not exist")
    img = cv2.resize(img,(224,224))
    img = np.reshape(img,[1,224,224,3])
    # Predict on data
    classes = model.predict(
        x=img,
        verbose=1,
    )

    labels = 'Blues', 'Classical', 'Country', 'Disco', 'Hiphop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock'
    sizes = classes[0]

    fig1, ax1 = plt.pyplot.subplots()
    ax1.pie(sizes, autopct='%1.1f%%', shadow=False, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.legend(labels, loc = 'best') 
    plt.pyplot.savefig('static/data/graph.png')

    
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(os.path.join('static/data/', secure_filename(f.filename)))
      predictor(secure_filename(f.filename))
      full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'graph.png')
      return render_template('results.html', user_image = full_filename)
        
if __name__ == '__main__':
   app.run(debug = True)
