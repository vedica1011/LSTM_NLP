from pickle import load
from numpy import array
from keras.models import load_model
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# generate a sequence of characters with a language model
def generate_seq(model, mapping, seq_length, seed_text, n_chars):
	in_text = seed_text
	# generate a fixed number of characters
	for _ in range(n_chars):
		# encode the characters as integers
		encoded = [mapping[char] for char in in_text]
		# truncate sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
		# one hot encode
		encoded = to_categorical(encoded, num_classes=len(mapping))
		# predict character
		#yhat = model.pre(encoded, verbose=0)
		predicted_probabilities = model.predict(encoded)  # x_test is your input data
		yhat = np.argmax(predicted_probabilities, axis=1)		
		# reverse map integer to character
		out_char = ''
		for char, index in mapping.items():
			if index == yhat:
				out_char = char
				break
		# append to input
		in_text += out_char
	return in_text

# load the model
model = load_model('model.h5')
# load the mapping
mapping = load(open('mapping.pkl', 'rb'))
# test start of rhyme
# print(generate_seq(model, mapping, 10, 'Sing a son', 20))
# # test mid-line
# print(generate_seq(model, mapping, 10, 'king was i', 20))
# # test not in original
# print(generate_seq(model, mapping, 10, 'hello worl', 20))

print(generate_seq(model, mapping, 10, "sing", 20))