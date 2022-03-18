from zipfile import ZipFile

# Download the dataset from the link below:
# https://www.kaggle.com/ashish2001/original-flickr8k-dataset

zf = ZipFile('./data/flickr8k-dataset.zip', 'r')
zf.extractall('./data')
zf.close()