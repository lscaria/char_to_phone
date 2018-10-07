import tensorflow as tf 
from models import test_model
from utils import PHONE_TO_ID, ID_TO_PHONE, CHAR_TO_ID, ID_TO_CHAR 

def word_to_token(word):
    results = []
    for letter in word:
        if letter in CHAR_TO_ID:
            results.append(CHAR_TO_ID[letter])
        else:
            results.append(CHAR_TO_ID['-'])
            
    return results

def token_to_phone(seq):
    results = []
    for val in seq:
        if val in ID_TO_PHONE:
            results.append(ID_TO_PHONE[val])
        else:
            results.append('~')
    return results
        

word = 'TENSOR'

tokens = word_to_token(word)
print(tokens)
word = ([tokens], len(tokens))
with tf.Graph().as_default() as g:
    logits = test_model([tokens],len(tokens), 1)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "../models/models.ckpt" )
        predictions = sess.run(logits)
        print(predictions)
        print(token_to_phone(predictions[0]))

