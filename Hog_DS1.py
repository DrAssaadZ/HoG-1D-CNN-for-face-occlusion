import cv2 as cv
import numpy as np
from os import path
import os
from sklearn.decomposition import PCA
from scipy.spatial import distance
from collections import Counter
from skimage.feature import hog
from skimage import exposure
import sklearn.metrics as metrics

# creating train and test folders
def prepare_dataset(dataset_path):
    # creating output folders
    os.mkdir('dataset')
    os.mkdir('dataset/training')
    os.mkdir('dataset/testing')
    output_folder = 'dataset/training/'
    counter = 0

    # listing dataset files
    dataset_files = os.listdir(dataset_path)

    for folder in dataset_files:

        # folder files [1.ppm, 10.ppm, 2.ppm ...]
        folder_files = os.listdir(dataset_path + folder)
        # folder files sorted [1.ppm, 2.ppm, 3.ppm ...]
        folder_files_sorted = sorted(folder_files, key=lambda x: int(os.path.splitext(x)[0]))

        # for each image in the
        for image in folder_files_sorted:
            if not path.exists(output_folder + folder[1:]):
                os.mkdir(output_folder + folder[1:])

            # reading then saving the image in its corresponding folder
            img = cv.imread(dataset_path + folder + '/' + image)
            cv.imwrite(output_folder + folder[1:] + '/' + image[:-4] + '.jpg', img)
            counter += 1

            if counter == 7:
                output_folder = 'dataset/testing/'

            # if counter == 12:
            #     folder = 'training/'

            if counter == 10:
                output_folder = 'dataset/training/'
                counter = 0


# takes an image and returns its histogram descriptor
def calculate_hog_descriptor(img):

    descriptor = hog(img, orientations=16, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), visualize=False, multichannel=False)
    return descriptor


def hog_recognition(train_dataset_path, test_dataset_path, knn, total_tr_imgs, total_tst_imgs, train_img_nbr_class, tst_img_nbr_class):
    # for train -------------------------------------------------------------------------------
    train_dataset_files = os.listdir(train_dataset_path)

	# sorting folders
    train_dataset_files_sorted = sorted(train_dataset_files, key=lambda x: int(os.path.splitext(x)[0]))

    # list that contains the images histogram descriptors
    train_images_hists = []
    # list that contains the image classes
    train_classes = []

    # for each train class folder
    for folder in train_dataset_files_sorted:

        # listing files inside each folder ( class )
        folder_files = os.listdir(train_dataset_path + folder)

        # for each image in the folder class
        for file in folder_files:
            # read image
            img = np.array(
                cv.imread(train_dataset_path + folder + '/' + file, cv.IMREAD_GRAYSCALE))  # read as grayscale
            # resized_img = cv.resize()

            # calculating the hog descriptor of the image
            img_hog = calculate_hog_descriptor(img)

            # storing the img descriptor in train image hists along with the image class
            train_images_hists.append(img_hog)
            train_classes.append(folder)

    # for test -------------------------------------------------------------------------------
    test_dataset_files = os.listdir(test_dataset_path)

	# sorting folders
    test_dataset_files_sorted = sorted(test_dataset_files, key=lambda x: int(os.path.splitext(x)[0]))

    test_images_distances = np.zeros(total_tr_imgs)

    # vector that contains the classes
    classes_vect = np.zeros(total_tst_imgs, dtype=int)

    current_class = 0

    # used for metrics (recall, precision ...)
    true_classes = []
    pre_classes = []

    # for each test class folder
    for folder in test_dataset_files_sorted:

        # listing files inside each folder ( class )
        folder_files = os.listdir(test_dataset_path + folder)

        current_image_index = 0

        # for each image in the folder class
        for file in folder_files:
            # read image
            img = cv.imread(test_dataset_path + folder + '/' + file, cv.IMREAD_GRAYSCALE)  # read as grayscale
            # resized_img = cv.resize()

            # calculating the hog descriptor of the image
            img_hog = calculate_hog_descriptor(img)

            # compraing each test images with all the train images
            for i in range(total_tr_imgs):
                query_distance = distance.euclidean(img_hog, train_images_hists[i])
                test_images_distances[i] = query_distance

            # getting the indexes of the smallest distances in the list
            smallest_indexes = sorted(range(len(test_images_distances)), key=lambda sub: test_images_distances[sub])[:knn]

            # calculating the classes from the smallest_indexes list
            img_possible_classes = list(map(lambda x: x // train_img_nbr_class, smallest_indexes))

            # most commun class in img_possible_classes
            image_class = Counter(img_possible_classes).most_common(1)
            print('predicted class :', image_class[0][0])
            print('real class :', current_class)

            # creating the predicted class vector (used for report (precision, recall ...))
            pre_classes.append(image_class[0][0])

            # if the predicted class is correct
            if int(image_class[0][0]) == current_class:
                classes_vect[current_class * tst_img_nbr_class + current_image_index] = 1

            current_image_index += 1

            # creating the true class vector (used for report (precision, recall ...))
            true_classes.append(current_class)

        current_class += 1

    # calculating accuracy
    accuracy_new = Counter(classes_vect).most_common(1)
    print(accuracy_new)
    if accuracy_new[0][0] == 0:
        print('accuracy', 1 - (accuracy_new[0][1] / total_tst_imgs))
    else:
        print('accuracy', accuracy_new[0][1] / total_tst_imgs)

    # calculating the metrics (precision, recall, f1 score)
    report = metrics.classification_report(true_classes, pre_classes)
    print(report)

# main ----------------------------------------------------------
orig_dataset_path = 'C:/Users/Ouss/Desktop/Datasets/faces dataset/att-database/Classes/'

# prepare_dataset(dataset_path=orig_dataset_path)

# train and test paths
train_path = 'dataset/training/'
test_path = 'dataset/testing/'

hog_recognition(train_dataset_path=train_path, test_dataset_path=test_path, knn=11, total_tr_imgs=280, total_tst_imgs=120, train_img_nbr_class=7, tst_img_nbr_class=3)