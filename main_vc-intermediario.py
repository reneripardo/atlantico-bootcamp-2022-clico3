import argparse
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from random import shuffle
from sklearn.model_selection import train_test_split
from sklearn import metrics,svm
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
from sklearn.preprocessing import MaxAbsScaler


def load_data(datadir, classes, img_size=100):
    training_data = []
    label = []
    for classe in range(len(classes)):
        path = os.path.join(datadir, classes[classe])
        shufled_list  = list(os.listdir(path))
        shuffle(shufled_list)
        for img in shufled_list:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            img_array = cv2.resize(img_array, (img_size, img_size))
            unique = np.unique(img_array)
            if len(unique) == 1:
                continue
            training_data.append(img_array)
            label.append(classe)
    return training_data , label




data , label = load_data('dataset/geometric',['circle','square','star','triangle'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--path_image', default=os.environ.get("PATH_IMAGE"),
                    help='path image example')
    ap.add_argument('-i2', '--path_image2', default=os.environ.get("PATH_IMAGE2"),
                    help='path image example')


    args = vars(ap.parse_args())

if __name__ == "__main__":
    main()