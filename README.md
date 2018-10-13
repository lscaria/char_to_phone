# char_to_phone

This repo contains a simple Seq2Seq model for character to phoneme conversion. 

## Preprocessing:
We use the [CMU pronouncation dictionary](http://www.speech.cs.cmu.edu/cgi-bin/cmudict) as the input dataset. All files can be found in the data folder.

We use TfRecords as the input to our seq2seq model. Each Record Example contains a word, the phoneme conversion, and the length of the conversion. 

To create the processed dataset, run the below line, which will create train, dev, and test splits.

```
python create_tfrecords.py
```

## Training:
The models we use for training are contained in models.py. When you run the training script, it runs the train and dev datasets for one epoch before printing out the loss and accuracy for both.

To train the model, run:
```
python train.py
```

The training, automatically updates the checkpoint file at the end of each epoch. Additionally, if you want to start training from a checkpoint, set line 61:
```
restore = True
```
otherwise it will start trianing a new model.


## Inference:
To test on your own data, run:
```
python test.py
```

As of now you have to manually update the word argument in line 25 to be the word you want the model to run inference on. This will change to a commandline interface soon.

## Acknowledgments:
I made this small project because I wanted a more clear understanding of seq2seq models and I couldn't finda tutorial which showed a simple version with different train, dev, and test graphs as the Tensorflow NMT tutorial shows. Additionally I wanted an updated tutorial which took advantage of the tf.data libaries. 

This small repo was inspired by the [Prononcing English Gradients](https://www.kaggle.com/reppic/predicting-english-pronunciations/notebook) kaggle notebook, with elements from the Tensorflow [NMT Tutorial](https://github.com/tensorflow/nmt/blob/master/nmt/model.py) and [Park Chansung's Medium](https://towardsdatascience.com/seq2seq-model-in-tensorflow-ec0c557e560f) post