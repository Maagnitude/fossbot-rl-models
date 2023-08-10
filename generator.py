import cv2
import skimage.measure
import numpy as np

def image_to_map(path:str, filter:int = 2):
    img = cv2.imread('sim_grid.png')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, im_bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    im_bw[im_bw == 255] = 1
    table = np.logical_not(im_bw).astype(int)
    #Maxpooling 
    table = skimage.measure.block_reduce(table, (filter,filter), np.max)
    return table