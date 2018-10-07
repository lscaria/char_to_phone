
VALID_ALPHABET =  "-ABCDEFGHIJKLMNOPQRSTUVWXYZ'"
START_PHONE = '\t'
END_PHONE = '\n'
PAD_PHONE= '\p'
cmu_dict_path = "../data/cmudict-0.7b.txt"
cmu_symbols_path = "../data/cmudict-0.7b.symbols.txt"
with open(cmu_symbols_path) as file:
	VALID_PHONES= [PAD_PHONE]+[START_PHONE] + [line.strip() for line in file] +[END_PHONE]
	print(len(VALID_PHONES))

# word_to_phone_dict = load_data()
# PHONE_TO_ID, ID_TO_PHONE = create_mapping(list(VALID_PHONES))
# CHAR_TO_ID, ID_TO_CHAR = create_mapping(list(VALID_ALPHABET))
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

word_to_phone_dict = load_data()
PHONE_TO_ID, ID_TO_PHONE = create_mapping(list(VALID_PHONES))
CHAR_TO_ID, ID_TO_CHAR = create_mapping(list(VALID_ALPHABET))


def is_correct(word, pronounciation):

	pronounciation = pronounciation.strip()

	correct_pronounciations = word_to_phone_dict[word]
	#print(correct_pronounciations, pronounciation)
	for correct_pronons in correct_pronounciations:
		if pronounciation == correct_pronons:
			#print('true')
			return True

	return False

def get_batch_accuracy(tokens, labels, preds):
	num_correct = 0 
	output = []
	for words, phonemes,prediction in zip(tokens, labels,preds):
	#print(words)
		word_seq = [ID_TO_CHAR[i] for i in words if i!=0]
		phone_seq =[ID_TO_PHONE[i] for i in phonemes if i!=0]
		pred_seq =[ID_TO_PHONE[i] for i in prediction if i!=0]
		output.append([word_seq, phone_seq, pred_seq])
		if is_correct("".join(word_seq), " ".join(pred_seq)):
			num_correct +=1
	return num_correct/len(tokens), output

if __name__ == '__main__':
	pr = ['F', 'L', 'IH1', 'T', 'IH1', 'SH', 'AH0', 'N', '\n']
	pro = " ".join(pr)
	print(pro)
	print(is_correct("FLIRTATION", ''))

