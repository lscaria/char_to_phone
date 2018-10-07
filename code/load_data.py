import tensorflow as tf 
import create_tfrecords 
from models import train_model, inference_model
from utils import is_correct
import numpy as np
VALID_ALPHABET = create_tfrecords.VALID_ALPHABET
VALID_PHONES = create_tfrecords.VALID_PHONES



def parse_function(tf_example):
	# Define how to parse the example
	context_features = {
	    "length": tf.FixedLenFeature([], dtype=tf.int64)
	}
	sequence_features = {
	    "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
	    "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64)
	}
	 
	# Parse the example
	context_parsed, sequence_parsed = tf.parse_single_sequence_example(
	    serialized=tf_example,
	    context_features=context_features,
	    sequence_features=sequence_features
	)

	return context_parsed['length'], sequence_parsed['tokens'], sequence_parsed['labels']

def create_iterator(filename, batch_size):

	dataset = tf.data.TFRecordDataset(filename)
	dataset = dataset.map(parse_function)
	dataset = dataset.padded_batch(batch_size, padded_shapes=([],[None],[None]))
	dataset = dataset.filter(lambda t,y,s: tf.equal(tf.shape(y)[0], batch_size))
	iterator = dataset.make_initializable_iterator()
	return iterator



char_to_id, id_to_char = create_tfrecords.create_mapping(list(VALID_ALPHABET))
phone_to_id, id_to_phone = create_tfrecords.create_mapping(list(VALID_PHONES))


def get_batch_accuracy(tokens, labels, preds):
	num_correct = 0 
	output = []
	for words, phonemes,prediction in zip(tokens, labels,preds):
	#print(words)
		word_seq = [id_to_char[i] for i in words if i!=0]
		phone_seq =[id_to_phone[i] for i in phonemes if i!=0]
		pred_seq =[id_to_phone[i] for i in prediction if i!=0]
		output.append([word_seq, phone_seq, pred_seq])
		if is_correct("".join(word_seq), " ".join(pred_seq)):
			num_correct +=1
	return num_correct/len(tokens), output


batch_size = 32
num_epochs = 200
restore = True

filename = tf.placeholder(tf.string, shape=[])
iterator = create_iterator(filename,batch_size)

length, token, label = iterator.get_next()


output = train_model(token,label, length, batch_size)
infer_output = inference_model(token, label, length, batch_size)
pred = tf.argmax(output, axis=2)

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=output)
cost = tf.reduce_mean(loss)
updates = tf.train.AdamOptimizer(1e-4).minimize(cost)

sess = tf.Session()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

if restore == True:
	saver.restore(sess, tf.train.latest_checkpoint('../models'))

for i in range(num_epochs):
	train_accuracy = []
	print("Epoch {}".format(i))

	sess.run(iterator.initializer, feed_dict={filename:'../data/processed/train.tfrecords'})
	while True:
		try:
			labels,tokens,preds,_,out_loss = sess.run([label,token,pred,updates,cost])
			accuracy, preds = get_batch_accuracy(tokens, labels, preds)
			#print(accuracy)
			train_accuracy.append(accuracy)
		except tf.errors.OutOfRangeError:
			print("Epoch:{}, Accuracy:{}".format(i, np.mean(train_accuracy)))
			break


	dev_accuracy = []
	sess.run(iterator.initializer, feed_dict={filename:'../data/processed/dev.tfrecords'})
	while True:
		try:
			labels, tokens,infered = sess.run([label, token,infer_output])
			accuracy, preds = get_batch_accuracy(tokens, labels, infered)
			#print(accuracy)
			dev_accuracy.append(accuracy)
		except tf.errors.OutOfRangeError:
			save_path = saver.save(sess, "../models/models.ckpt")
			print("Epoch:{}, Dev Accuracy:{}".format(i, np.mean(dev_accuracy)))
			break




# with tf.Session() as sess:
# 	sess.run(tf.global_variables_initializer())
# 	train_filename = ['../data/processed/train.tfrecords']
# 	sess.run(iterator.initializer,feed_dict={filename:train_filename})

	
# 	for i in range(100000):
# 		labels,tokens,preds,_,out_loss = sess.run([label,token,pred,updates,cost])


# 		if i%100==0:

# 			accuracy, preds = get_batch_accuracy(tokens, labels, preds)
# 			print("Loss", out_loss)
# 			print("Accuracy", accuracy)

# 			if i%1000==0:
# 				labels, tokens,infered = sess.run([label, token,infer_output])
# 				print('Dev Accuracy = ')
# 				accuracy, preds = get_batch_accuracy(tokens, labels, infered)
# 				print("infered accuracy", accuracy)


# 		if i ==99999:
# 			accuracy, preds = get_batch_accuracy(tokens, labels, preds)
# 			print("Loss", out_loss)
# 			print("Accuracy", accuracy)
			
		