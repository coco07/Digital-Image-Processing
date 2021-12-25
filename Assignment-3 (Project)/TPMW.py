# USAGE: TPMW(image_name, class_).
# does our proposed Three Perturbations Multiple Window blind forensic technique on image_name of class
# class (-1 for pristine or 1 for median filtered image) and saves the feature vector in TPMW_feat_vec.txt
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


def TPMW(image_name, class_):
    # first, the image is read
    image = cv2.imread(image_name)

    # When doing 3 perturbations with 4 different window sizes and calculating 8 different
    # image quality measures per perturbation, the final feature vector of IQMs has 3x4x8=96 dimensions.
    feat_vec = np.zeros(97)

    # the extra dimension here (the first) is for the class of the sample. If you are training the classifier,
    # the class must be known. If you are testing the sample, you can put 0 here.
    feat_vec[0] = class_

    # Now, we are going to start the perturbations!
    # First, the image is progressively blurred three times with a 3x3 window
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

    # Now, the image is progressively blurred three times with a 5x5 window
    # at each blurry, the blurred image is compared to the input image by means of image quality metrics.
    # we use eight image quality metrics in the comparison

    blurred_image_1_5x5 = cv2.medianBlur(image, 5)
    temp_vec = calculate_IQMs(image, blurred_image_1_5x5)
    feat_vec[25] = temp_vec[0]
    feat_vec[26] = temp_vec[1]
    feat_vec[27] = temp_vec[2]
    feat_vec[28] = temp_vec[3]
    feat_vec[29] = temp_vec[4]
    feat_vec[30] = temp_vec[5]
    feat_vec[31] = temp_vec[6]
    feat_vec[32] = temp_vec[7]

    blurred_image_2_5x5 = cv2.medianBlur(blurred_image_1_5x5, 5)
    temp_vec = calculate_IQMs(image, blurred_image_2_5x5)
    feat_vec[33] = temp_vec[0]
    feat_vec[34] = temp_vec[1]
    feat_vec[35] = temp_vec[2]
    feat_vec[36] = temp_vec[3]
    feat_vec[37] = temp_vec[4]
    feat_vec[38] = temp_vec[5]
    feat_vec[39] = temp_vec[6]
    feat_vec[40] = temp_vec[7]

    blurred_image_3_5x5 = cv2.medianBlur(blurred_image_2_5x5, 5)
    temp_vec = calculate_IQMs(image, blurred_image_3_5x5)
    feat_vec[41] = temp_vec[0]
    feat_vec[42] = temp_vec[1]
    feat_vec[43] = temp_vec[2]
    feat_vec[44] = temp_vec[3]
    feat_vec[45] = temp_vec[4]
    feat_vec[46] = temp_vec[5]
    feat_vec[47] = temp_vec[6]
    feat_vec[48] = temp_vec[7]

    # Now, the image is progressively blurred three times with a 7x7 window
    # at each blurry, the blurred image is compared to the input image by means of image quality metrics.
    # we use eight image quality metrics in the comparison

    blurred_image_1_7x7 = cv2.medianBlur(image, 7)
    temp_vec = calculate_IQMs(image, blurred_image_1_7x7)
    feat_vec[49] = temp_vec[0]
    feat_vec[50] = temp_vec[1]
    feat_vec[51] = temp_vec[2]
    feat_vec[52] = temp_vec[3]
    feat_vec[53] = temp_vec[4]
    feat_vec[54] = temp_vec[5]
    feat_vec[55] = temp_vec[6]
    feat_vec[56] = temp_vec[7]

    blurred_image_2_7x7 = cv2.medianBlur(blurred_image_1_7x7, 7)
    temp_vec = calculate_IQMs(image, blurred_image_2_7x7)
    feat_vec[57] = temp_vec[0]
    feat_vec[58] = temp_vec[1]
    feat_vec[59] = temp_vec[2]
    feat_vec[60] = temp_vec[3]
    feat_vec[61] = temp_vec[4]
    feat_vec[62] = temp_vec[5]
    feat_vec[63] = temp_vec[6]
    feat_vec[64] = temp_vec[7]

    blurred_image_3_7x7 = cv2.medianBlur(blurred_image_2_7x7, 7)
    temp_vec = calculate_IQMs(image, blurred_image_3_7x7)
    feat_vec[65] = temp_vec[0]
    feat_vec[66] = temp_vec[1]
    feat_vec[67] = temp_vec[2]
    feat_vec[68] = temp_vec[3]
    feat_vec[69] = temp_vec[4]
    feat_vec[70] = temp_vec[5]
    feat_vec[71] = temp_vec[6]
    feat_vec[72] = temp_vec[7]

    # Now, the image is progressively blurred three times with a 9x9 window
    # at each blurry, the blurred image is compared to the input image by means of image quality metrics.
    # we use eight image quality metrics in the comparison

    blurred_image_1_9x9 = cv2.medianBlur(image, 9)
    temp_vec = calculate_IQMs(image, blurred_image_1_9x9)
    feat_vec[73] = temp_vec[0]
    feat_vec[74] = temp_vec[1]
    feat_vec[75] = temp_vec[2]
    feat_vec[76] = temp_vec[3]
    feat_vec[77] = temp_vec[4]
    feat_vec[78] = temp_vec[5]
    feat_vec[79] = temp_vec[6]
    feat_vec[80] = temp_vec[7]

    blurred_image_2_9x9 = cv2.medianBlur(blurred_image_1_9x9, 9)
    temp_vec = calculate_IQMs(image, blurred_image_2_9x9)
    feat_vec[81] = temp_vec[0]
    feat_vec[82] = temp_vec[1]
    feat_vec[83] = temp_vec[2]
    feat_vec[84] = temp_vec[3]
    feat_vec[85] = temp_vec[4]
    feat_vec[86] = temp_vec[5]
    feat_vec[87] = temp_vec[6]
    feat_vec[88] = temp_vec[7]

    blurred_image_3_9x9 = cv2.medianBlur(blurred_image_2_9x9, 9)
    temp_vec = calculate_IQMs(image, blurred_image_3_9x9)
    feat_vec[89] = temp_vec[0]
    feat_vec[90] = temp_vec[1]
    feat_vec[91] = temp_vec[2]
    feat_vec[92] = temp_vec[3]
    feat_vec[93] = temp_vec[4]
    feat_vec[94] = temp_vec[5]
    feat_vec[95] = temp_vec[6]
    feat_vec[96] = temp_vec[7]

    # np.savetxt(feat_vec_file_name, feat_vec, fmt="%.2f")
    np.set_printoptions(precision=2, suppress=True)
    return feat_vec


if __name__ == '__main__':

    cv_img = []

    # generating feature vector
    for img in glob.glob("./dataset/dataset_pr/*.tif"):
        n = TPMW(img, -1)
        cv_img.append(n)
    for img in glob.glob("./dataset/dataset_mf/*.tif"):
        n = TPMW(img, 1)
        cv_img.append(n)

    # converting list to pandas data frame
    df = pd.DataFrame(cv_img)

    # saving feature vector into text file
    np_array = df.to_numpy()
    np.savetxt("TPMW_feat_vec.txt", np_array, fmt="%0.2f")
    print("feature vector saved in TPMW_feat_vec.txt file...")

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