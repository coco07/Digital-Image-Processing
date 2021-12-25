# USAGE: TPOW(image_name, class_).
# does our proposed Three Perturbations One Window blind forensic technique on image_name of class
# class (-1 for pristine or 1 for median filtered image) and saves the feature vector in TPOW_feat_vec.txt
# If you are training the classifier, the argument class must be known and correctly given by you.

import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, train_test_split

from IQMS.calculate_IQMs import calculate_IQMs


def TPOW(image_name, class_):
    # first, the image is read
    image = cv2.imread(image_name)

    # When doing 3 perturbations with only 1 window size and calculating 8 different
    # image quality measures per perturbation, the final feature vector of IQMs has 3x1x8=24 dimensions.
    feat_vec = np.zeros(25)

    # the extra dimension here (the first) is for the class of the sample. If you are training the classifier,
    # the class must be known. If you are testing the sample, you can put 0 here.
    feat_vec[0] = class_

    # Now, we are going to start the perturbations!
    # The image is progressively blurred three times with a 3x3 window
    # at each blurry, the blurred image is compared to the input image by means of image quality metrics.
    # we use eight image quality metrics in the comparison

    blurred_image_1_3x3 = cv2.medianBlur(image, 3)
    temp_vec = calculate_IQMs(image, blurred_image_1_3x3)
    feat_vec[1] = temp_vec[0]
    feat_vec[2] = temp_vec[1]
    feat_vec[3] = temp_vec[2]
    feat_vec[4] = temp_vec[3]
    feat_vec[5] = temp_vec[4]
    feat_vec[6] = temp_vec[5]
    feat_vec[7] = temp_vec[6]
    feat_vec[8] = temp_vec[7]

    blurred_image_2_3x3 = cv2.medianBlur(blurred_image_1_3x3, 3)
    temp_vec = calculate_IQMs(image, blurred_image_2_3x3)
    feat_vec[9] = temp_vec[0]
    feat_vec[10] = temp_vec[1]
    feat_vec[11] = temp_vec[2]
    feat_vec[12] = temp_vec[3]
    feat_vec[13] = temp_vec[4]
    feat_vec[14] = temp_vec[5]
    feat_vec[15] = temp_vec[6]
    feat_vec[16] = temp_vec[7]

    blurred_image_3_3x3 = cv2.medianBlur(blurred_image_2_3x3, 3)
    temp_vec = calculate_IQMs(image, blurred_image_3_3x3)
    feat_vec[17] = temp_vec[0]
    feat_vec[18] = temp_vec[1]
    feat_vec[19] = temp_vec[2]
    feat_vec[20] = temp_vec[3]
    feat_vec[21] = temp_vec[4]
    feat_vec[22] = temp_vec[5]
    feat_vec[23] = temp_vec[6]
    feat_vec[24] = temp_vec[7]


    # np.savetxt(feat_vec_file_name, feat_vec, fmt="%.2f")
    np.set_printoptions(precision=2, suppress=True)
    return feat_vec


if __name__ == '__main__':

    cv_img = []

    # generating feature vector
    for img in glob.glob("./dataset/dataset_pr/*.tif"):
        n = TPOW(img, -1)
        cv_img.append(n)
    for img in glob.glob("./dataset/dataset_mf/*.tif"):
        n = TPOW(img, 1)
        cv_img.append(n)

    # converting list to pandas data frame
    df = pd.DataFrame(cv_img)

    # saving feature vector into text file
    np_array = df.to_numpy()
    np.savetxt("TPOW_feat_vec.txt", np_array, fmt="%0.2f")
    print("feature vector saved in TPOW_feat_vec.txt file...")

    features = df.iloc[:, 1:]
    labels = df.iloc[:, :1]
    # print(features, labels)

    d_train, d_test, l_train, l_test = train_test_split(features, labels, test_size=0.3)

    # basic SVM model
    # model = SVC(kernel = 'rbf', probability=True)
    # model.fit(d_train, l_train.values.ravel())
    # print(accuracy_score(model.predict(d_test), l_test))

    # Define the parameter grid for C from 0.001 to 10, gamma from 0.001 to 10
    C_grid = [0.001, 0.01, 0.1, 1, 10]
    gamma_grid = [0.001, 0.01, 0.1, 1, 10]
    param_grid = {'C': C_grid, 'gamma': gamma_grid}

    print("\nFinding Optimal Model Parameter using inbuilt Grid-Search Method of SVM 'rbf' kernel: ")
    grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=3, scoring="accuracy")
    grid.fit(d_train, l_train.values.ravel())

    # Find the best model parameters
    print(f"Best Grid Score: {grid.best_score_}")
    print(f"Best parameters: {grid.best_params_}")
    print(f"Best estimator function: {grid.best_estimator_}")

    # Optimal SVM model
    model = SVC(kernel='rbf', C=0.001, gamma=0.001, probability=True)
    model.fit(d_train, l_train.values.ravel())
    print(f"\nAfter applying Best parameters, Model metrics are: ")

    predicted_label = model.predict(d_test)

    cm1 = confusion_matrix(l_test, predicted_label)
    print('Confusion Matrix : \n', cm1)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=['1 \n(MF)', '-1 \n(Pristine)'])
    disp.plot()
    plt.show()

    total1 = sum(sum(cm1))
    accuracy1 = round((cm1[0, 0] + cm1[1, 1]) / total1, 2)
    print('Accuracy : ', accuracy1)

    sensitivity1 = round(cm1[0, 0] / (cm1[0, 0] + cm1[0, 1]), 2)
    print('Sensitivity (or Recall Score) : ', sensitivity1)

    specificity1 = round(cm1[1, 1] / (cm1[1, 0] + cm1[1, 1]), 2) if (cm1[1, 0] + cm1[1, 1]) else 0
    print('Specificity : ', specificity1)

    precision1 = round(cm1[0, 0] / (cm1[0, 0] + cm1[1, 0]), 2)
    print('Precision Score : ', precision1)