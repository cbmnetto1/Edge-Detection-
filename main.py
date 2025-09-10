#open cv code
#https://docs.opencv.org/3.4/da/d5c/tutorial_canny_detector.html

from __future__ import print_function
import cv2 as cv
import numpy as np

img = cv.imread('lena.png', cv.IMREAD_GRAYSCALE)
if img is None:
    print('Could not open or find the image')
    exit(0)

#open cv canny
canny = cv.Canny(img, threshold1=100, threshold2=200)

def compute_gradients(img):
    Hx = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])       # Horizontal
    Hy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])        # Vertical
    H45 = np.array([[-2,-1,0],[-1,0,1],[0,1,2]])       # 45°
    H135 = np.array([[0,1,2],[-1,0,1],[-2,-1,0]])      # 135°

    Mx = cv.filter2D(img, cv.CV_64F, Hx)
    My = cv.filter2D(img, cv.CV_64F, Hy)
    M45 = cv.filter2D(img, cv.CV_64F, H45)
    M135 = cv.filter2D(img, cv.CV_64F, H135)

    magnitude = np.sqrt(Mx**2 + My**2 + M45**2 + M135**2)
    magnitude = cv.convertScaleAbs(magnitude)

    return magnitude

def ImprovedCanny(src_gray):
    # --- 1. Filtro contra ruído (median blur no lugar do Gaussian) ---
    img_blur = cv.medianBlur(src_gray, 3)

    # --- 2. Calcular gradientes em 4 direções ---
    mag = compute_gradients(img_blur)

    # --- 3. Threshold adaptativo com Otsu ---
    _, edge_map = cv.threshold(mag, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    return edge_map

edges = ImprovedCanny(img)

cv.imshow('OG', img)
cv.imshow('canny', canny)
cv.imshow('improved canny', edges)
cv.waitKey(0)
cv.destroyAllWindows()