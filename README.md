```
__        _______ _     ____ ___  __  __ _____   _____ ___
\ \      / / ____| |   / ___/ _ \|  \/  | ____| |_   _/ _ \ 
 \ \ /\ / /|  _| | |  | |  | | | | |\/| |  _|     | || | | |
  \ V  V / | |___| |__| |__| |_| | |  | | |___    | || |_| |
   \_/\_/  |_____|_____\____\___/|_|  |_|_____|   |_| \___/

 _____ _   _ _____
|_   _| | | | ____|
  | | | |_| |  _|
  | | |  _  | |___
  |_| |_| |_|_____|

 _____ _   _ _   _ _   _ ____  _____ ____  ____   ___  __  __ _____
|_   _| | | | | | | \ | |  _ \| ____|  _ \|  _ \ / _ \|  \/  | ____|
  | | | |_| | | | |  \| | | | |  _| | |_) | | | | | | | |\/| |  _|
  | | |  _  | |_| | |\  | |_| | |___|  _ <| |_| | |_| | |  | | |___
  |_| |_| |_|\___/|_| \_|____/|_____|_| \_\____/ \___/|_|  |_|_____|
```
```
OVERVIEW:
This project contains a number of deliverables:

## CNN music genre classifier ##

  Files: 
    main.py - main file that runs commands and reads tags
    preprocess.py - processes data in preparation for training and testing
    model.py - contains model
    hyperparameters.py - contains hyperparameters
  Directories: 
    test - holds testing data
    train - holds training data
    weights - holds model weights from training
    logs - holds logs from trainging
    predict - holds audio files for prediction feature
  How to run:
    To train this CNN, simply execute 'main.py'
    To train using pre-existing weights, execute 'main.py --load-checkpoint <path to weights>'
    To evaluate on test data, execute 'main.py --evaluate --load-checkpoint <path to weights>'
    To classify an audio file and view pie chart of prediction results, place an audio file 
    (wav, mp3, mp4) in the 'predict' directory and execute 'main.py --predict' The model will 
    automatically load the highest available weights in the 'weights' directory

## genre prediction Flask app ##
  Files:
    app.py - script that runs website
  Directories:
    templates - contains html templates for website pages
    static - contains css file as well as location for audio files that are downloaded 
    from site during operation
  How to run:
    Run app.py and click on link to website. Once on the website, you can upload any audio 
    file and the site will load a prediction based on the best model weights available in 
    'weights'. The audio file and pie chart are stored in 'static/data`

## audio to spectrogram conversion script ##
  Files:
    convert_spectrogram.py - converts audio files to spectrograms and builds file structure 
    for the model data
    download.sh - downloads the audio files into proper folders from kaggle
    run.sh - runs download.sh and convert spectrogram
    asshole.svg - no idea, please forgive us
    sleep.sh - no clue
  How to run: **NOTE: this only needs to be run once to prepare data and traning environment, 
              it does not need to be run since the project is initialized**
    Run 'download.sh' once downloading had completed, run 'convert_spectrogram.py'


```