from utils import *
from pickle import dump
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

# Clean image descriptions
filename = './data/Flickr8k.token.txt'
descriptions = read_image_descriptions(filename)
print(f"Total number of descriptions: {len(descriptions)}")
clean_description_text(descriptions)
save_description(descriptions, 'iv3_desc.txt')
vocabulary = create_vocabulary(descriptions)
print(f"The size of vocabulary is: {len(vocabulary)}")

model = InceptionV3()

#  Feature extraction from images
directory = './data/Flickr8k_Dataset'
features = get_image_features(directory, model, preprocess_input)
print("The length of all the extracted features is:", len(features))

# Save all the features to pickle file
dump(features, open('./iv3_feats.pkl', 'wb'))

# Load training dataset (6000 out of 2000) as present in Flickr_8k.trainImages.txt
filename = './data/Flickr_8k.trainImages.txt'
train_id = get_identifiers_set(filename)
train_desc = get_clean_descriptions('./iv3_desc.txt', train_id)
train_features = load_image_features('./iv3_feats.pkl', train_id)
train_desc['1000268201_693b08cb0e']

print(f"The length of train identifiers is: {len(train_id)}")
print(f"The length of train descriptions is: {len(train_desc)}")
print(f"The length of train features is: {len(train_features)}")

tokenizer = create_tokenizer(train_desc)
dump(tokenizer, open('./iv3_tokenizer.pkl', 'wb'))
vocab_size = len(tokenizer.word_index) + 1
max_length = max_lengths(train_desc)
print(f"The size of the vocabulary is: {vocab_size}")
print(f"The maximum length of train descriptions is: {max_length}")


# Train model and save our model as .h5 file
model = create_model(vocab_size, max_length)
epochs = 10
steps = len(train_desc)

for i in range(epochs):
    print(f"Epoch: {i + 1}")
    generator = data_generator(train_desc, train_features, tokenizer, max_length, vocab_size)
    model.fit_generator(generator=generator, epochs=1, steps_per_epoch=steps, verbose=1)
    
model.save(f"./models/iv3.h5" )