import tensorflow as tf 
import create_tfrecords 
VALID_ALPHABET = create_tfrecords.VALID_ALPHABET
VALID_PHONES = create_tfrecords.VALID_PHONES
START_PHONE = create_tfrecords.START_PHONE

def process_decoder_input(target_data, batch_size):
	#batch_size = 32
	go_id = tf.cast(VALID_PHONES.index(START_PHONE), tf.int64)
	after_slice = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
	print(after_slice.dtype)
	after_concat = tf.concat( [tf.fill([batch_size, 1], go_id), after_slice], 1)
	return after_concat

def basic_model(x,y):
	x_one_hot = tf.one_hot(x, 27)
	y_one_hot = tf.one_hot(y, 86)


	with tf.variable_scope('encoder'):
		encode_lstm = tf.contrib.rnn.LSTMCell(num_units=256)
		
		#hidden_state = tf.zeros([32])
		#current_state = tf.zeros([32])
		state = encode_lstm.zero_state(32, dtype=tf.float32)
		encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(encode_lstm, x_one_hot,initial_state=state, dtype=tf.float32)

	with tf.variable_scope('decoder'):
		decode_lstm = tf.contrib.rnn.LSTMCell(num_units=256)
		decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(decode_lstm, y_one_hot,initial_state=encoder_final_state, dtype=tf.float32)

	out = tf.layers.dense(decoder_outputs,86, activation=tf.nn.softmax)
	#output, state = lstm(x_one_hot, state)
	return out




def encoder_model(x,y, seqlen, batch_size):
	y = process_decoder_input(y, batch_size)
	print(y.shape)

	src_vocab_size = 28
	target_vocab_size=87
	embedding_size = 50


	with tf.variable_scope('encoder'):
		embedding_encoder = tf.get_variable(
    "embedding_encoder", [src_vocab_size, embedding_size])


		encoder_emb_inp = tf.nn.embedding_lookup(embedding_encoder, x)
		encode_lstm = tf.contrib.rnn.LSTMCell(num_units=256)
		
		#hidden_state = tf.zeros([32])
		#current_state = tf.zeros([32])
		state = encode_lstm.zero_state(batch_size, dtype=tf.float32)
		encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(encode_lstm, encoder_emb_inp, initial_state=state)

	dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, 50]))
	dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, y)
	with tf.variable_scope('decoder'):
		decode_lstm = tf.contrib.rnn.LSTMCell(num_units=256)

		output_layer = tf.layers.Dense(target_vocab_size)
		seqlen = tf.cast(seqlen, dtype=tf.int32)
		helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input, seqlen )

		decoder = tf.contrib.seq2seq.BasicDecoder(decode_lstm, helper, encoder_final_state,output_layer)

		outputs, _,_ = tf.contrib.seq2seq.dynamic_decode(decoder)

		return outputs.rnn_output

		#decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(decode_lstm, y_one_hot,initial_state=encoder_final_state, dtype=tf.float32)

	#out = tf.layers.dense(decoder_outputs,86, activation=tf.nn.softmax)
	#output, state = lstm(x_one_hot, state)
	return out

