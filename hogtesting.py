import cv2 as cv
import numpy as np
from os import path
import os
from sklearn.decomposition import PCA
from scipy.spatial import distance
from collections import Counter
from skimage.feature import hog
from skimage import exposure
import shutil
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold
import matplotlib.pyplot as plt


# getting the image class, function used in the kfold cross validation method
def get_image_class(img_name):
    file_split = img_name.split('-')
    img_class = file_split[0]
    return img_class


# creating train and test folders
def prepare_dataset(dataset_path, nbr_fold):
    os.mkdir('dataset3/')

    # listing images in the dataset path
    img_list = np.array(os.listdir(dataset_path))

    # list containing the classes
    img_classes = []
    for image in img_list:
        img_classes.append(get_image_class(image))

    cross_val = StratifiedKFold(n_splits=nbr_fold,  shuffle=True, random_state=101)

    # creating the folds folders
    for i in range(nbr_fold):
        os.mkdir('dataset3/fold ' + str(i))

    current_fold = 0

    # for each fold we get train and test list of indexes
    for train_index_list, test_index_list in cross_val.split(img_list, img_classes):

        # creating the train and test folders inside the folds
        fold_train_path = 'dataset3/fold ' + str(current_fold) + '/training/'
        os.mkdir(fold_train_path)
        fold_test_path = 'dataset3/fold ' + str(current_fold) + '/testing/'
        os.mkdir(fold_test_path)

        # for each train dataset index (obtained from the kfold method )
        for train_index in train_index_list:
            # making the class folder if it doesn't exist
            if not path.exists(fold_train_path + img_classes[train_index]):
                os.mkdir(fold_train_path + img_classes[train_index])
            # copying the image into the corresponding class
            shutil.copy(dataset_path + img_list[train_index],
                        fold_train_path + img_classes[train_index] + '/' + img_list[train_index])

        # for each test dataset index (obtained from the kfold method )
        for test_index in test_index_list:
            # making the class folder if it doesn't exist
            if not path.exists(fold_test_path + img_classes[test_index]):
                os.mkdir(fold_test_path + img_classes[test_index])
            # copying the image into the corresponding class
            shutil.copy(dataset_path + img_list[test_index],
                        fold_test_path + img_classes[test_index] + '/' + img_list[test_index])

        print('current_fold : ', current_fold)
        current_fold += 1



# takes an image and returns its histogram descriptor
def calculate_hog_descriptor(img, orien):

    img = cv.resize(img, (120, 165))
    descriptor = hog(img, orientations=orien, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), visualize=False, multichannel=False)
    return descriptor


def hog_recognition(train_dataset_path, test_dataset_path, knn, total_tr_imgs, total_tst_imgs, train_img_nbr_class, tst_img_nbr_class, orien):
    # for train -------------------------------------------------------------------------------
    train_dataset_files = os.listdir(train_dataset_path)

    # list that contains the images histogram descriptors
    train_images_hists = []
    # list that contains the image classes
    train_classes = []

    # for each train class folder
    for folder in train_dataset_files:

        # listing files inside each folder ( class )
        folder_files = os.listdir(train_dataset_path + folder)

        # for each image in the folder class
        for file in folder_files:
            # read image
            img = np.array(
                cv.imread(train_dataset_path + folder + '/' + file, cv.IMREAD_GRAYSCALE))  # read as grayscale
            # resized_img = cv.resize()

            # calculating the hog descriptor of the image
            img_hog = calculate_hog_descriptor(img, orien)

            # storing the img descriptor in train image hists along with the image class
            train_images_hists.append(img_hog)
            train_classes.append(folder)
        break    

    # for test -------------------------------------------------------------------------------
    test_dataset_files = os.listdir(test_dataset_path)

    test_images_distances = np.zeros(total_tr_imgs)

    # vector that contains the classes
    classes_vect = np.zeros(total_tst_imgs, dtype=int)

    current_class = 0

    # for each test class folder
    for folder in test_dataset_files:

        # listing files inside each folder ( class )
        folder_files = os.listdir(test_dataset_path + folder)

        current_image_index = 0

        # for each image in the folder class
        for file in folder_files:
            # read image
            img = cv.imread(test_dataset_path + folder + '/' + file, cv.IMREAD_GRAYSCALE)  # read as grayscale
            # resized_img = cv.resize()

            # calculating the hog descriptor of the image
            img_hog = calculate_hog_descriptor(img, orien)

            
            print('test histogram: ', len(img_hog))
            # plt.hist(img_hog, bins = 17024)
            # plt.show()
            # compraing each test images with all the train images
            for i in range(14):
                print('train histograms :', train_images_hists[i])
                query_distance = distance.euclidean(img_hog, train_images_hists[i])
                test_images_distances[i] = query_distance
            return
            # print('distances:', test_images_distances)
            # getting the indexes of the smallest distances in the list
            smallest_indexes = sorted(range(len(test_images_distances)), key=lambda sub: test_images_distances[sub])[:knn]

            # calculating the classes from the smallest_indexes list
            img_possible_classes = list(map(lambda x: x // train_img_nbr_class, smallest_indexes))
            print('possible ', img_possible_classes)
            # most commun class in img_possible_classes
            image_class = Counter(img_possible_classes).most_common(1)
            print('predicted class :', image_class[0][0])
            print('real class :', current_class)

            # if the predicted class is correct
            if int(image_class[0][0]) == current_class:
                classes_vect[current_class * tst_img_nbr_class + current_image_index] = 1

            current_image_index += 1

        current_class += 1
    print('classes vect', classes_vect)
    # calculating accuracy
    accuracy_new = Counter(classes_vect).most_common(1)
    # print(accuracy_new)
    if accuracy_new[0][0] == 0:
        print('accuracy (', orien, ',', knn, ') = ' , 1 - (accuracy_new[0][1] / total_tst_imgs))
    else:
        print('accuracy (', orien, ',', knn, ') = ' , accuracy_new[0][1] / total_tst_imgs)


# main ----------------------------------------------------------
orig_dataset_path = 'E:/Master/Datasets/faces dataset/AR face dataset/cropped Dataset/original cropped images/'

# prepare_dataset(dataset_path=orig_dataset_path, nbr_fold=5)

# train and test paths
train_path = 'dataset3_robustness/training/'
test_path = 'dataset3_robustness/testing/'

orientations = [9]
k = [7]

for i in orientations:
    for j in k:
        hog_recognition(train_dataset_path=train_path, test_dataset_path=test_path, knn=j, total_tr_imgs=1400, 
                total_tst_imgs=1200, train_img_nbr_class=14, tst_img_nbr_class=12, orien=i)