import os
import pickle
import sys

from img2vec_pytorch import Img2Vec
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split


def train(data_dir):
    img2vec = Img2Vec()



    data = {}
    # collect all image features and labels from data_dir, then split into train/val
    features = []
    labels = []
    
    for category in os.listdir(data_dir):
        cat_dir = os.path.join(data_dir, category)
        if not os.path.isdir(cat_dir):
            continue
        for img_name in os.listdir(cat_dir):
            img_path = os.path.join(cat_dir, img_name)
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    vec = img2vec.get_vec(img)
                features.append(vec)
                labels.append(category)
                print(f"label: {category}")
            except Exception as e:
                print(f"Skipping {img_path}: {e}")


    if len(features) == 0:
        print("No images found in", data_dir)
        data['training_data'] = np.array([])
        data['training_labels'] = np.array([])
        data['validation_data'] = np.array([])
        data['validation_labels'] = np.array([])
    else:
        strat = labels if len(set(labels)) > 1 else None
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=strat
        )
        data['training_data'] = X_train
        data['training_labels'] = y_train
        data['validation_data'] = X_val
        data['validation_labels'] = y_val

    # # train model

    model = RandomForestClassifier(random_state=0)
    
    model.fit(data['training_data'], data['training_labels'])


    # test performance
    y_pred = model.predict(data['validation_data'])
    score = accuracy_score(y_pred, data['validation_labels'])

    print(score)

    # # save the model
    with open('./model2.p', 'wb') as f:
        pickle.dump(model, f)
        f.close()
        
    
def main():
    if len(sys.argv) != 2:
        print("Usage: python train.py <data_directory>")
        sys.exit(1)

    data_directory = sys.argv[1]
    train(data_directory)


if __name__ == '__main__':
    main()