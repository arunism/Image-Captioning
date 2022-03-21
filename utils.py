import os
import string
import numpy as np
from tqdm import tqdm
from pickle import load
from tensorflow.keras.layers import Add
from tensorflow.keras.models import Model
from nltk.translate.bleu_score import corpus_bleu
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Input, LSTM, Dropout, Embedding, Dense


def read_image_descriptions(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()

    image_id_dict = dict()
    for line in text.split('\n'):
        tokens = line.split()
        if len(line) < 2:
            continue
        image_id, image_description = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        image_description = ' '.join(image_description)
        if image_id not in image_id_dict:
            image_id_dict[image_id] = list()
        image_id_dict[image_id].append(image_description)
    return image_id_dict


def clean_description_text(description):
    translator = str.maketrans('', '', string.punctuation)
    
    for key, desc_list in description.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            desc = desc.split()
            desc = [word.lower() for word in desc]
            desc = [word.translate(translator) for word in desc]
            desc = [word for word in desc if len(word)>1]
            desc = [word for word in desc if word.isalpha()]
            desc_list[i] = ' '.join(desc)


def save_description(description, filename):
    lines = list()
    for key, desc_list in description.items():
        for desc in desc_list:
            lines.append(key + " " + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


def get_image_features(directory, model, preprocess_input):
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

    features = dict()
    for file_name in tqdm(os.listdir(directory)):
        filename = f"{directory}/{file_name}"
        image = load_img(filename, target_size=(299, 299))
        image = img_to_array(image)
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        image_id = file_name.split('.')[0]
        features[image_id] = feature
    return features


def load_image_features(filename, dataset):
    all_features = load(open(filename, 'rb'))
    features = {k: all_features[k] for k in dataset}
    return features


def create_vocabulary(description):
    all_desc = set()
    for key in description.keys():
        [all_desc.update(desc.split()) for desc in description[key]]
    return all_desc


def get_identifiers_set(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    
    identifiers = list()
    for line in text.split('\n'):
        if len(line) < 1:
            continue
        image_id = line.split('.')[0]
        identifiers.append(image_id)
    return set(identifiers)


def get_clean_descriptions(filename, identifiers):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    
    descriptions = dict()
    for line in text.split('\n'):
        tokens = line.split()
        image_id, image_desc = tokens[0], tokens[1:]
        if image_id in identifiers:
            if image_id not in descriptions:
                descriptions[image_id] = list()
            desc = "starttoken " + ' '.join(image_desc) + " endtoken"
            descriptions[image_id].append(desc)
    return descriptions


def to_list(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(one_of_five_desc) for one_of_five_desc in descriptions[key]]
    return all_desc


def create_tokenizer(descriptions):
    desc_list = to_list(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(desc_list)
    return tokenizer


def max_lengths(descriptions):
    desc_list = to_list(descriptions)
    return max(len(desc.split()) for desc in desc_list)


def create_encoded_sequence(tokenizer, max_length, desc_list, image_feat, vocab_size):
    image_features, image_desc, next_words = list(), list(), list()
    
    for desc in desc_list:
        seq = tokenizer.texts_to_sequences([desc])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            image_features.append(image_feat)
            image_desc.append(in_seq)
            next_words.append(out_seq)
    return np.array(image_features), np.array(image_desc), np.array(next_words)


def create_model(vocab_size, max_length):
    input1 = Input(shape=(max_length,))
    seq1 = Embedding(vocab_size, 256, mask_zero=True)(input1)
    seq2 = Dropout(0.5)(seq1)
    seq3 = LSTM(256)(seq2)
    
    # Feature extractor model
    input2 = Input(shape=(2048,))
    feat1 = Dropout(0.5)(input2)
    feat2 = Dense(256, activation='relu')(feat1)
    
    # Decoder model
    decoder1 = Add()([feat2, seq3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    output = Dense(vocab_size, activation='softmax')(decoder2)
    
    # Tie all the image features and word sequence together using keras Model class
    model = Model(inputs=[input2, input1], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model


def data_generator(descriptions, image_feats, tokenizer, max_length, vocab_size):
    while 1:
        for key, desc_list in descriptions.items():
            image_feat = image_feats[key][0]
            image_features, desc, next_words = create_encoded_sequence(tokenizer, max_length, desc_list, image_feat, vocab_size)
            yield ([image_features, desc], next_words)


def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def generate_desc(model, tokenizer, image_feat, max_length):
    in_text = 'starttoken'

    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image_feat, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endtoken':
            break
    return in_text


def evaluate_model(model, descriptions, image_feat, tokenizer, max_length):
    reference, candidate = list(), list()
    # Iterate for the entire set of images
    for key, desc_list in tqdm(descriptions.items()):
        yhat = generate_desc(model, tokenizer, image_feat[key], max_length)
        references = [desc.split() for desc in desc_list]
        reference.append(references)
        candidate.append(yhat.split())
        
    print(f"Cumulative 1-gram: {corpus_bleu(reference, candidate, weights=(1,0,0,0))}")
    print(f"Cumulative 2-gram: {corpus_bleu(reference, candidate, weights=(0.5,0.5,0,0))}")
    print(f"Cumulative 3-gram: {corpus_bleu(reference, candidate, weights=(0.33,0.33,0.33,0))}")
    print(f"Cumulative 4-gram: {corpus_bleu(reference, candidate, weights=(0.25,0.25,0.25,0.25))}")


def extract_test_features(filename, model, preprocess_input):
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    image = load_img(filename, target_size=(299, 299))
    image = img_to_array(image)
    image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature