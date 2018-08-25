import tensorflow as tf 
import create_tfrecords 
from models import basic_model
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



def load_tfrecords():

	filename = ['../data/processed/train.tfrecords']

	dataset = tf.data.TFRecordDataset(filename)
	dataset = dataset.map(parse_function)
	dataset = dataset.padded_batch(32, padded_shapes=([],[None],[None]))
	dataset = dataset.filter(lambda t,y,s: tf.equal(tf.shape(t)[0], 32))
	dataset = dataset.repeat()
	iterator = dataset.make_one_shot_iterator()

	return iterator.get_next()



length, token, label = load_tfrecords()
labels_onehot = tf.one_hot(label, 86)
#seq_in = tf.reshape(token, [32,-1])
#seq_in = tf.placeholder(tf.float32,[None])
output = basic_model(token,label)
pred = tf.argmax(output, axis=2)
loss = tf.losses.mean_squared_error(labels=label, predictions=pred)
cost = tf.reduce_mean(loss)
updates = tf.train.AdamOptimizer(1e-4).minimize(cost)
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(1000):
		labels,tokens,preds,_,out_loss = sess.run([label,token,pred,updates,cost])
		#print('ouptut' ,labels.shape)
		#print(preds)
		print(out_loss)

		if i ==999:
			char_to_id, id_to_char = create_tfrecords.create_mapping(list(VALID_ALPHABET))
			phone_to_id, id_to_phone = create_tfrecords.create_mapping(list(VALID_PHONES))
			for words, phonemes,prediction in zip(tokens, labels,preds):
				#print(words)
				word_seq = [id_to_char[i] for i in words if i!=0]
				phone_seq =[id_to_phone[i] for i in phonemes if i!=0]
				pred_seq =[id_to_phone[i] for i in prediction if i!=0]
				print(word_seq, phone_seq,pred_seq)
		