import cv2
import numpy as np

def process_image(img_path):
    """
    Function that preprocessing the image for modelling or predicting later
    """

    # Read image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Crop image
    rows = (img != 0).any(axis=1)
    cols = (img != 0).any(axis=0)
    img = img[rows, :][:, cols]
    
    # Padding with 0s
    num_rows, num_cols = img.shape
    add_rows = num_rows // 6
    add_cols = num_cols // 6
    
    img = np.r_[np.zeros((add_rows, num_cols)), img, np.zeros((add_rows, num_cols))]
    new_rows = img.shape[0]
    img = np.c_[np.zeros((new_rows, add_cols)), img, np.zeros((new_rows, add_cols))]
    
    # Resize
    roi = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA) / 255
    
    return roi