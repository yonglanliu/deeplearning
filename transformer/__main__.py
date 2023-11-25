START_TOKEN = "<SOS>"
PADDING_TOKEN = "<PADDING>"
END_TOKEN = "<EOS>"

chinese_vocabulary = [START_TOKEN, ]
english_vocabulary = [START_TOKEN, ' ', '!', '"', '#', 'S']

index_to_chinese = {k: v for k, v in enumerate(chinese_vocabulary)}
chinese_to_index = {v: k for k, v in enumerate(chinese_vocabulary)}

index_to_english = {k: v for k, v in enumerate(english_vocabulary)}
english_to_index = {v: k for k, v in enumerate(english_vocabulary)}

with open(english_file) as file:
	english_sentences = file.readlines()
with open(chinese_file) as file:
	chinese_sentences = file.readlines()

TOTAL_SENTENCES = 100000
english_sentences = english_sentences[:TOTAL_SENTENCES]
chinese_sentences = chinese_sentences[:TOTAL_SENTENCES]

english_sentences = [sentence.rstrip('\n') for sentence in english_sentences]
chinese_sentences = [sentence.rstrip('\n') for sentence in chinese_sentences]


PERCENTILE = 97
print(f'{PERCENTILE}th percentile length Chinese: {np.percentile([len(x) for x in chinese_sentences], PERCENTILE)}')
print(f'{PERCENTILE}th percentile length English: {np.percentile([len(x) for x in english_sentences], PERCENTILE)}')

max_sequence_length = 200
def is_valid_tokens(sentence, vocab):
	for token in list(set(sentence)):
		if token not in vocab:
			return False
		return True

def is_valid_length(sentence, max_sequence_length):
	return len(list(sentence)) < (max_sequence_length - 1) # need to re_add the end token so leaving 1 space

valid_sentence_indices = []
for index in range(len(chinese_sentences)):
	chinese_sentence, english_sentence = chinese_sentences[index], english_sentences[index]
	if is_valid_length(chinese_sentence, max_sequence_length) \
	and is_valid_length(english_sentence, max_sequence_length) \
	and is_valid_tokens(chinese_sentence, chinese_vocabulary):
		valid_sentence_indices.append(index)

print(f"Number of sentences: {len(chinese_sentences)}")
print(f"Number of valid sentences: {len(valid_sentence_indices)}")

from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
	def __init__(self, english_sentences, chinese_sentences):
		self.english_sentences = english_sentences
		self.chinese_sentences = chinese_sentences

	def __len__(self):
		return len(self.english_sentences)

	def __getitem__(self, idx):
		return self.english_sentences[idx], self.chinese_sentences[idx]

dataset = TextDataset(english_sentences, chinese_sentences)

batch_size = 3
train_loader = DataLoader(dataset, batch_sizes)
iterator = iter(train_loader)

for batch_num, batch in enumerate(iterator):
	print(batch)
	if batch_num > 3:
		break

# add start, end, and padding tokens
def tokenize(sentence, language_to_index, start_token=True, end_token=True):
	sentence_word_indicies = [language_to_index[token] for token in list(sentence)]
	if start_token:
		sentence_word_indicies.insert(0, language_to_index[START_TOKEN])
	if end_token:
		sentence_word_indicies.insert(-1, language_to_index[End_TOKEN])
	for _ in range(len(sentence_word_indicies), max_sequence_length):
		sentence_word_indicies.append(language_to_index[PADDING_TOKEN])
	return torch.tensor(sentence_to_indicies)


eng_tokenized, cn_tokenized = [], []
for sentence_num in range(batch_size):
	eng_sentence, cn_sentence = batch[0][sentence_num], batch[1][sentence_num]
	eng_tokenized.append(tokenize(eng_sentence, english_to_index, start_token=False, end_token=False))
	cn_tokenized.append(tokenize(cn_sentence, chinese_to_index, start_token=True, end_token=False)

eng_tokenized = torch.stack(eng_tokenized)
cn_tokenized = torch.stack(cn_tokenized)