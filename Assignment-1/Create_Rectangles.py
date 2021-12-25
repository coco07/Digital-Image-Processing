# Digital Image Processing
# Assignment-I (Question-2)
# Group-10

import cv2
import numpy as np

# Function to validate if the new rectangle to be drawn is overlapping with existing rectangles or not
def valid_rectangle(image_bg, M_total, N_total, height, width, border, Vb, rect_orientation, corner_i, corner_j):
    flag1 = True
    if rect_orientation == 1:   # landscape
        for j in range(corner_j, corner_j + width - 1):
            for i in range(corner_i, corner_i + height - 1):
                if image_bg[j][i] >= Vb[0]:
                    pass
                else:
                    flag1 = False
        if flag1 == True:
            return corner_i, corner_j
        else:
            corner_i = int(np.random.uniform(border, N_total - border - 1 - height))
            corner_j = int(np.random.uniform(border, M_total - border - 1 - width))
            # Recursive call to the function to find out non-overlapping space for the new rectangle (with orientation=1)
            return valid_rectangle(image_bg, M_total, N_total, height, width, border, Vb, rect_orientation, corner_i,
                                   corner_j)
    elif rect_orientation == 2:     # portrait
        for j in range(corner_j, corner_j + height - 1):
            for i in range(corner_i, corner_i + width - 1):
                if image_bg[j][i] >= Vb[0]:
                    pass
                else:
                    flag1 = False
        if flag1 == True:
            return corner_i, corner_j
        else:
            corner_i = int(np.random.uniform(border, N_total - border - 1 - width))
            corner_j = int(np.random.uniform(border, M_total - border - 1 - height))
            return valid_rectangle(image_bg, M_total, N_total, height, width, border, Vb, rect_orientation, corner_i,
                                   corner_j)

# Function to draw valid rectangles and create final image
def create_rectangles(M, N, border, n, w1, w2, alpha, orientation, Vf=[0], Vb=[255]):
    M_total = M + 2 * (border)
    N_total = N + 2 * (border)

    if Vf[0] == 0 and Vb[0] == 255:
        gray_scale = False
        image_bg = Vb * np.ones(shape=[M_total, N_total], dtype=np.uint8)  # Canvas
        image_bg[:, :border] = image_bg[:border, :] = image_bg[:, (N_total - border):] = image_bg[(M_total - border):,:] = 0  # Border

    else:
        gray_scale = True
        image_bg = np.random.randint(Vb[0], Vb[1], (M_total, N_total), dtype=np.uint8)
        image_bg[:, :border] = image_bg[:border, :] = image_bg[:, (N_total - border):] = image_bg[(M_total - border):,:] = 0

    for rect in range(n):
        if gray_scale == True:
            intensity_fg = int(np.random.uniform(Vf[0], Vf[1]))
        else:
            intensity_fg = 0

        rect_orientation = np.random.choice(orientation)

        width = int(np.random.uniform(w1, w2))
        height = width * alpha

        try:
            if rect_orientation == 1:
                corner_i = int(np.random.uniform(border, N_total - border - 1 - height))
                corner_j = int(np.random.uniform(border, M_total - border - 1 - width))
                corner_i, corner_j = valid_rectangle(image_bg, M_total, N_total, height, width, border, Vb,
                                                     rect_orientation, corner_i, corner_j)
                for j in range(corner_j, corner_j + width - 1):
                    for i in range(corner_i, corner_i + height - 1):
                        image_bg[j][i] = intensity_fg
            if rect_orientation == 2:
                corner_i = int(np.random.uniform(border, N_total - border - 1 - width))
                corner_j = int(np.random.uniform(border, M_total - border - 1 - height))
                corner_i, corner_j = valid_rectangle(image_bg, M_total, N_total, height, width, border, Vb,
                                                     rect_orientation, corner_i, corner_j)
                for j in range(corner_j, corner_j + height - 1):
                    for i in range(corner_i, corner_i + width - 1):
                        image_bg[j][i] = intensity_fg
        except RecursionError:
            return create_rectangles(2 * M, 2 * N, border, n, w1, w2, alpha, orientation, Vf, Vb)

    # cv2.imshow("final_image", image_bg)          # Show image
    cv2.imwrite('final_image.jpg', image_bg)       # Save Image into the same location that of the program
    print(f'final shape of the image: {image_bg.shape}')

# Main function             # parameter description
def main():
    M = 300                 # no of rows
    N = 200                 # no of columns
    border = 5              # size of the border
    n = 50                  # number of rectangles to fit
    w1 = 20                 # lower bound for the width of rectangle
    w2 = 30                 # upper bound for the width of rectangle
    alpha = 2               # fixed [height,width] ratio of rectangle
    orientation = [1, 2]    # uniformly distributed orientation of rectangle [landscape=1, portrait=2]
    Vf = [0, 128]           # foreground colours distributed uniformly, if not provided then default = 0
    Vb = [129, 255]         # background colours distributed uniformly, if not provided then default = 255

    # Function call with Vf and Vb as optional parameters
    create_rectangles(M, N, border, n, w1, w2, alpha, orientation, Vf, Vb)
    # create_rectangles(M, N, border, n, w1, w2, alpha, orientation)

# Main function call
if __name__ == '__main__':
    main()