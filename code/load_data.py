import tensorflow as tf 
import create_tfrecords 
from models import basic_model, encoder_model, inference_model
from utils import is_correct
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



def load_tfrecords(filename, batch_size):

	#filename = ['../data/processed/train.tfrecords']

	dataset = tf.data.TFRecordDataset(filename)
	dataset = dataset.map(parse_function)
	dataset = dataset.padded_batch(batch_size, padded_shapes=([],[None],[None]))
	dataset = dataset.filter(lambda t,y,s: tf.equal(tf.shape(y)[0], batch_size))
	dataset = dataset.repeat()
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
		#print(word_seq, phone_seq,pred_seq)
		output.append([word_seq, phone_seq, pred_seq])
		if is_correct("".join(word_seq), " ".join(pred_seq)):
			num_correct +=1
	return num_correct/len(tokens), output

	#return iterator.get_next()
batch_size = 32

filename = tf.placeholder(tf.string, shape=[None])
iterator = load_tfrecords(filename,32)
#length, token, label = load_tfrecords(filename)
length, token, label = iterator.get_next()
labels_onehot = tf.one_hot(label, 86)
#seq_in = tf.reshape(token, [32,-1])
#seq_in = tf.placeholder(tf.float32,[None])
output = encoder_model(token,label, length, batch_size)
infer_output = inference_model(token, label, length, batch_size)
pred = tf.argmax(output, axis=2)
#infer_pred = tf.argmax(infer_output, axis=2)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=output)
cost = tf.reduce_mean(loss)
updates = tf.train.AdamOptimizer(1e-4).minimize(cost)
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	train_filename = ['../data/processed/train.tfrecords']
	sess.run(iterator.initializer,feed_dict={filename:train_filename})

	
	for i in range(100000):
		labels,tokens,preds,_,out_loss = sess.run([label,token,pred,updates,cost])
		#print('ouptut' ,labels.shape)
		#print(preds)
		#print(out_loss)

		if i%100==0:

			accuracy, preds = get_batch_accuracy(tokens, labels, preds)
			print("Loss", out_loss)
			print("Accuracy", accuracy)

			if i%1000==0:
				infered = sess.run(infer_output)
				print('Dev Accuracy = ')
				accuracy, preds = get_batch_accuracy(tokens, labels, infered)
				print("infered accuracy", accuracy)
				#print("infered preds", preds)

				#print(accuracy)
				#print(preds)

		if i ==99999:
			accuracy, preds = get_batch_accuracy(tokens, labels, preds)
			print("Loss", out_loss)
			print("Accuracy", accuracy)
			
		