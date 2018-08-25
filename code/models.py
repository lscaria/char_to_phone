import tensorflow as tf 


def basic_model(x,y):
	x_one_hot = tf.one_hot(x, 27)
	y_one_hot = tf.one_hot(y, 86)


	with tf.variable_scope('encoder'):
		encode_lstm = tf.contrib.rnn.LSTMCell(num_units=32)
		
		#hidden_state = tf.zeros([32])
		#current_state = tf.zeros([32])
		state = encode_lstm.zero_state(32, dtype=tf.float32)
		encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(encode_lstm, x_one_hot,initial_state=state, dtype=tf.float32)

	with tf.variable_scope('decoder'):
		decode_lstm = tf.contrib.rnn.LSTMCell(num_units=32)
		decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(decode_lstm, y_one_hot,initial_state=encoder_final_state, dtype=tf.float32)

	out = tf.layers.dense(decoder_outputs,86, activation=tf.nn.softmax)
	#output, state = lstm(x_one_hot, state)
	return out