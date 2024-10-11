
import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
import librosa # uncomment for prediction
import librosa.display # uncomment for prediction
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from model import ClassifierModel
import hyperparameters as hp
from preprocess import Datasets

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def parse_args():
    """ Perform command-line argument parsing. """
    parser = argparse.ArgumentParser(description="Let's train some neural nets!")
    parser.add_argument(
        '--load-checkpoint',
        default=None,
        help='''Path to model checkpoint file (should end with the
        extension .h5). Checkpoints are automatically saved when you
        train your model. If you want to continue training from where
        you left off, this is how you would load your weights. In
        the case of task 2, passing a checkpoint path will disable
        the loading of VGG weights.'''),
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="""Skips training and evaluates on the test set once.""",
    ),
    parser.add_argument(
        '--predict',
        help="Don't need to load weights, automatically finds best weights to use for prediction",
        action='store_true',)

    return parser.parse_args()


def train(model, datasets):
    """ Training routine. """
    if not os.path.exists('weights'):
        os.makedirs('weights')

    checkpoint_path = "./weights/"

    # Keras callbacks for training
    callback_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path + \
                    "weights.e{epoch:02d}-" + \
                    "acc{val_sparse_categorical_accuracy:.4f}.h5",
            monitor='val_sparse_categorical_accuracy',
            save_best_only=True,
            save_weights_only=True),
        tf.keras.callbacks.TensorBoard(
            update_freq='batch',
            profile_batch=0)
    ]
    # Begin training
    model.fit(
        x=datasets.train_data,
        validation_data=datasets.test_data,
        epochs=hp.num_epochs,
        batch_size=None,
        callbacks=callback_list,
    )


def test(model, test_data):
    """ Testing routine. """

    # Run model on test set
    model.evaluate(
        x=test_data, verbose=1,
    )

def predict(model, dataset):
    """ Predicting routine. Predicts genre of first file that is .mp4 or .mp3 """
    # find first file that contains .mp4 or .mp3 in predict dir
    mypath = './predict'
    files = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    files = list(filter(lambda x: (x[-4:] == '.mp3') or (x[-4:] == '.mp4') or (x[-4:] == '.wav'), files))
    if len(files) == 0:
        raise ValueError("No .mp4 or .mp3 files to predict")

    # converting sound to png
    full_name = os.path.join('./predict', files[0])
    x, sr = librosa.load(full_name)
    S = librosa.feature.melspectrogram(x)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB)
    plt.tight_layout()
    plt.savefig('./predict/test.png')
    plt.close()

    # loading png
    path = os.path.join('./predict', 'test.png')
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

    # making pie chart
    labels = 'Blues', 'Classical', 'Country', 'Disco', 'Hiphop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock'
    sizes = classes[0]

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, autopct='%1.1f%%', shadow=False, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1. legend(labels, loc = 'best') 
    plt.show()
    


def main():
    """ Main function. """
    # setting up dir for prediction if folder doesn't exist
    if not os.path.exists('predict'):
        os.makedirs('predict')

    datasets = Datasets(".")

    model = ClassifierModel()
    model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
    model.summary()

    # load weights for training
    if (ARGS.load_checkpoint is not None):
        model.load_weights(ARGS.load_checkpoint)

    # load best weight for prediction
    if predict:
        mypath = './weights'
        # getting best weight
        files = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
        files = list(filter((lambda x: 'weights' in x), files))
        if len(files) == 0:
            raise ValueError("No weights")

        files.sort(reverse=True, key=lambda x: int(x[17:21]))
        model.load_weights(os.path.join(mypath, files[0]))

    # Compile model graph
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"],
    )
   
    if ARGS.predict:
        predict(model, datasets)
    elif ARGS.evaluate:
        test(model, datasets.test_data)
    else:
        train(model, datasets)


# Make arguments global
ARGS = parse_args()

main()
