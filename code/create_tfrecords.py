import tensorflow as tf


VALID_ALPHABET =  "0ABCDEFGHIJKLMNOPQRSTUVWXYZ'"
START_PHONE = '\t'
END_PHONE = '\n'
PAD_PHONE= '\p'
cmu_dict_path = "../data/cmudict-0.7b.txt"
cmu_symbols_path = "../data/cmudict-0.7b.symbols.txt"
with open(cmu_symbols_path) as file:
	VALID_PHONES= [PAD_PHONE]+[START_PHONE] + [line.strip() for line in file] +[END_PHONE]
	print(len(VALID_PHONES))


def valid_word(word):
	for char in word:
		if char not in VALID_ALPHABET:
			return False
	return True
def load_data():

	word_to_phone_dict = {}

	with open(cmu_dict_path, encoding = "ISO-8859-1") as words_dict:
		for line in words_dict:
			if line[0] != ";":

				if line[0] not in VALID_ALPHABET:
					#print(line)
					continue
				word, phones = line.split("  ")

				if not valid_word(word):
					continue

				#remove '/n'
				phones = phones[:-1]
				
				#word has multiple pronounciations
				if "(" in word:
					word = word[:-3]
					word_to_phone_dict[word].append(phones)
					#print(word_to_phone_dict[word])
				else:
					word_to_phone_dict[word]=[phones]

	return word_to_phone_dict


def create_mapping(chars):
    char_to_id = {c: i for i, c in enumerate(chars)} 
    id_to_char = {i: c for i, c in enumerate(chars)}
    return char_to_id, id_to_char

def make_example(sequence, labels):
    # The object we return
    ex = tf.train.SequenceExample()
    # A non-sequential feature of our example
    sequence_length = len(labels)
    ex.context.feature["length"].int64_list.value.append(sequence_length)
    # Feature lists for the two sequential features of our example
    fl_tokens = ex.feature_lists.feature_list["tokens"]
    fl_labels = ex.feature_lists.feature_list["labels"]
    for token in sequence:
        fl_tokens.feature.add().int64_list.value.append(token)
    for label in labels:
        fl_labels.feature.add().int64_list.value.append(label)
    return ex

def create_tfrecord(split_list, wordtophone, split):

	char_to_id, id_to_char = create_mapping(list(VALID_ALPHABET))
	phone_to_id, id_to_phone = create_mapping(list(VALID_PHONES))

	writer = tf.python_io.TFRecordWriter('../data/processed/'+split+'.tfrecords')

	for word in split_list:
		#print(word)
		word_seq = [char_to_id[char] for char in word]
		#print(word_seq)

		for pronounciation in wordtophone[word]:
			
			pronounciation = pronounciation 
			#print(pronounciation)
			pr_seq = [phone_to_id[phone] for phone in pronounciation.split(' ')]
			pr_seq.append(phone_to_id[END_PHONE])

			ex = make_example(word_seq, pr_seq)
			writer.write(ex.SerializeToString())
	writer.close()


def main():
	wordtophone = load_data()
	print(len(wordtophone))


	keys = list(wordtophone.keys())
	keys = keys[0:1000]

	size = len(keys)
	train_set = keys[:int(size*0.8)]
	dev_set = keys[int(size*0.8):int(size*0.9)]
	test_set = keys[int(size*0.9):]

	print(size, len(train_set), len(test_set), len(dev_set))

	write_dir = '../data/processed'
	if tf.gfile.Exists(write_dir):
		tf.gfile.DeleteRecursively(write_dir)
	tf.gfile.MakeDirs(write_dir)

	create_tfrecord(train_set, wordtophone,'train')
	create_tfrecord(dev_set, wordtophone,'dev')
	create_tfrecord(test_set, wordtophone,'test')


if __name__ == '__main__':
	main()