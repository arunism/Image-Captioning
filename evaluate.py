from pickle import load
from tensorflow.keras.models import load_model
from utils import generate_desc, extract_test_features


def generate_caption(model):
    if model == 'iv3':
        from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
        model_instance = InceptionV3()
    elif model == 'vgg16':
        from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
        model_instance = VGG16()
    else:
        print("Only VGG16 and InceptionV3 models are supported!")
        exit()
    max_length = 34
    my_model = load_model(f'./models/{model}.h5')
    tokenizer = load(open(f'./{model}_tokenizer.pkl', 'rb'))
    test_image = extract_test_features('./images/cycle.jpg', model_instance, preprocess_input)
    description = generate_desc(my_model, tokenizer, test_image, max_length)
    # print(description)
    
    # Remove start and end tokens
    query = description
    stopwords = ['starttoken', 'endtoken']
    query_words = query.split()
    result = [word for word in query_words if word not in stopwords]
    result = ' '.join(result)
    print(result)

generate_caption('iv3')
# generate_caption('vgg16')