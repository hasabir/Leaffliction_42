import pickle
import sys

from img2vec_pytorch import Img2Vec
from PIL import Image


with open('./model2.p', 'rb') as f:
    model = pickle.load(f)

img2vec = Img2Vec()

image_path = sys.argv[1]  # Get image path from command line argument

img = Image.open(image_path)

features = img2vec.get_vec(img)

pred = model.predict([features])

print(pred)