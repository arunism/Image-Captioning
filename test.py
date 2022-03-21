from pickle import load
from tensorflow.keras.models import load_model
from utils import get_identifiers_set, get_clean_descriptions, load_image_features, evaluate_model

def test(model):
    max_length = 34
    tokenizer = load(open(f'./{model}_tokenizer.pkl', 'rb'))
    filename_test = './data/Flickr_8k.testImages.txt'
    test_id = get_identifiers_set(filename_test)
    test_desc = get_clean_descriptions(f'./{model}_desc.txt', test_id)
    test_features = load_image_features(f'./{model}_feats.pkl', test_id)

    print(f"The length of test identifiers is: {len(test_id)}")
    print(f"The length of test descriptions is: {len(test_desc)}")
    print(f"The length of test features is: {len(test_desc)}")

    filename = f'./models/{model}.h5'
    model = load_model(filename)

    # Evaluate model
    evaluate_model(model, test_desc, test_features, tokenizer, max_length)
    

# test('vgg16')
test('iv3')