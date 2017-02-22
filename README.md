# LSTM + CTC + Tensorflow Example

This is a demo using lstm and ctc to recognize a picture of  a series numbers with blanks all at once.

For example:given the piture below the model would give result `73791096754314441539`.

![](https://raw.githubusercontent.com/synckey/tensorflow_lstm_ctc_ocr/20160731-save-the-model/00000007_73791096754314441539_1.png)


## Installation
```
pip install opencv-python
brew install cmake
brew tap homebrew/science
brew install opencv
sh ./prepare_train_data.sh
```
The `prepare_train_data.sh` script would download the [SUN database](http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz) and extract the pitures to bgs dir. Then you can run `python gen.py` to generate test and train dir.

When the train and test data set are ready you can start the train process by `nohup python lstm_and_ctc_ocr_train.py `.

## Requirements

- Python 2.7+
- Tensorflow 1.0+
- python_speech_features
- numpy
- scipy

##
## License

This project is licensed under the terms of the MIT license.

See README for more information.
