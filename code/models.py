import tensorflow as tf 
import create_tfrecords 
VALID_ALPHABET = create_tfrecords.VALID_ALPHABET
VALID_PHONES = create_tfrecords.VALID_PHONES
START_PHONE = create_tfrecords.START_PHONE
END_PHONE = create_tfrecords.END_PHONE


def process_decoder_input(target_data, batch_size):

	go_id = tf.cast(VALID_PHONES.index(START_PHONE), tf.int64)
	after_slice = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
	print(after_slice.dtype)
	after_concat = tf.concat( [tf.fill([batch_size, 1], go_id), target_data], 1)
	return after_concat


def train_model(x,y, seqlen, batch_size):
	prepad = y
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
		state = encode_lstm.zero_state(batch_size, dtype=tf.float32)
		encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(encode_lstm, encoder_emb_inp, initial_state=state)

	with tf.variable_scope('decoder'):
		dec_embeddings = tf.get_variable("dec_embeddings",[target_vocab_size, 50])
		dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, y)
		decode_lstm = tf.contrib.rnn.LSTMCell(num_units=256)

		output_layer = tf.layers.Dense(target_vocab_size)
		seqlen = tf.cast(seqlen, dtype=tf.int32)
		helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input, seqlen )

		decoder = tf.contrib.seq2seq.BasicDecoder(decode_lstm, helper, encoder_final_state,output_layer)

		outputs, _,_ = tf.contrib.seq2seq.dynamic_decode(decoder)

		return outputs.rnn_output


def inference_model(x,y, seqlen, batch_size):
	#y = process_decoder_input(y, batch_size)
	print(y.shape)

	src_vocab_size = 28
	target_vocab_size=87
	embedding_size = 50

	with tf.variable_scope('encoder', reuse= True):
		embedding_encoder = tf.get_variable(
    "embedding_encoder", [src_vocab_size, embedding_size])

		encoder_emb_inp = tf.nn.embedding_lookup(embedding_encoder, x)
		encode_lstm = tf.contrib.rnn.LSTMCell(num_units=256)
		state = encode_lstm.zero_state(batch_size, dtype=tf.float32)
		encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(encode_lstm, encoder_emb_inp, initial_state=state)


	with tf.variable_scope('decoder', reuse = True):
		dec_embeddings = tf.get_variable("dec_embeddings",[target_vocab_size, 50])
		dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, y)
		maximum_iterations = tf.round(tf.reduce_max(seqlen) * 2)
		maximum_iterations = tf.cast(maximum_iterations, tf.int32)
		decode_lstm = tf.contrib.rnn.LSTMCell(num_units=256)

		output_layer = tf.layers.Dense(target_vocab_size)
		seqlen = tf.cast(seqlen, dtype=tf.int32)
		helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings,
			tf.fill([batch_size], VALID_PHONES.index(START_PHONE)), VALID_PHONES.index(END_PHONE))

		decoder = tf.contrib.seq2seq.BasicDecoder(decode_lstm, helper, encoder_final_state,output_layer)

		outputs, _,_ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=maximum_iterations)

		return outputs.sample_id


def test_model(x, seqlen,batch_size):


	src_vocab_size = 28
	target_vocab_size=87
	embedding_size = 50

	with tf.variable_scope('encoder'):
		embedding_encoder = tf.get_variable(
    "embedding_encoder", [src_vocab_size, embedding_size])

		encoder_emb_inp = tf.nn.embedding_lookup(embedding_encoder, x)
		encode_lstm = tf.contrib.rnn.LSTMCell(num_units=256)
		state = encode_lstm.zero_state(batch_size, dtype=tf.float32)
		encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(encode_lstm, encoder_emb_inp, initial_state=state)


	with tf.variable_scope('decoder'):
		dec_embeddings = tf.get_variable("dec_embeddings",[target_vocab_size, 50])
		#dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, y)
		maximum_iterations = tf.round(tf.reduce_max(seqlen) * 2)
		maximum_iterations = tf.cast(maximum_iterations, tf.int32)
		decode_lstm = tf.contrib.rnn.LSTMCell(num_units=256)

		output_layer = tf.layers.Dense(target_vocab_size)
		seqlen = tf.cast(seqlen, dtype=tf.int32)
		helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings,
			tf.fill([batch_size], VALID_PHONES.index(START_PHONE)), VALID_PHONES.index(END_PHONE))

		decoder = tf.contrib.seq2seq.BasicDecoder(decode_lstm, helper, encoder_final_state,output_layer)

		outputs, _,_ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=maximum_iterations)

		return outputs.sample_id
