import numpy as np
import cv2
import copy
import  math
from math import *

def cropImage(img,degree,pt1,pt2,pt3,pt4):
    height,width=img.shape[:2]
    matRotation=cv2.getRotationMatrix2D((width/2,height/2),degree,1)
    imgRotation = cv2.warpAffine(img, matRotation, (width, height), borderValue=(255, 255, 255))
    pt1 = list(pt1)
    pt3 = list(pt3)
    [[pt1[0]], [pt1[1]]] = np.dot(matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))

    if int(pt1[1]) > (pt3[1]):
        imgOut = imgRotation[int(pt3[1]):int(pt1[1]), int(pt3[0]):int(pt1[0])]
    else:
        imgOut = imgRotation[int(pt1[1]):int(pt3[1]), int(pt1[0]):int(pt3[0])]
    return imgOut

def rotated_coordinate(coordinate_x, coordinate_y, sita, cx, cy):
    input = np.array([coordinate_x, coordinate_y])
    matrix = np.array([[math.cos(sita), -math.sin(sita)], [math.sin(sita), math.cos(sita)]])
    x, y = np.dot(matrix, input.T)
    x = x + cx
    y = y + cy
    return x, y

def get_crack(img, bbox,offset):
    cx = bbox[0]
    cy = bbox[1]
    w = bbox[2]
    h = bbox[3]
    arc = w*h
    sita = bbox[4]


    left = -w/2
    top = -h/2
    right = w/2
    bottom = h/2

    x1, y1 = rotated_coordinate(left, top, sita, cx, cy)
    x2, y2 = rotated_coordinate(left, bottom, sita, cx, cy)
    x3, y3 = rotated_coordinate(right, bottom, sita, cx, cy)
    x4, y4 = rotated_coordinate(right, top, sita, cx, cy, )

    pt1 = (int(x1+offset), int(y1+offset))
    pt2 = (int(x2+offset), int(y2+offset))
    pt3 = (int(x3+offset), int(y3+offset))
    pt4 = (int(x4+offset), int(y4+offset))


    if sita > pi / 2:
        imgout = cropImage(img, -degrees(pi - sita), pt1, pt2, pt3, pt4)
    else:
        imgout = cropImage(img, degrees(sita), pt1, pt2, pt3, pt4)

    return imgout, arc

def rotateImage(img,degree,pt1,pt3,roi):
    height,width=img.shape[:2]
    matRotation=cv2.getRotationMatrix2D((width/2,height/2), degree, 1)
    imgRotation = cv2.warpAffine(img, matRotation, (width, height), borderValue=(255, 255, 255), flags=cv2.INTER_LANCZOS4)
    pt1 = list(pt1)
    pt3 = list(pt3)

    [[pt1[0]], [pt1[1]]] = np.dot(matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))

    if int(pt1[1]) > (pt3[1]):
        imgRotation[int(pt3[1]):int(pt1[1]), int(pt3[0]):int(pt1[0])] = roi

    else:
        imgRotation[int(pt1[1]):int(pt3[1]), int(pt1[0]):int(pt3[0])] = roi

    matRotation_ver = cv2.getRotationMatrix2D((width / 2, height / 2), -degree, 1)
    imgout = cv2.warpAffine(imgRotation,matRotation_ver,(width, height), borderValue=(255, 255, 255), flags=cv2.INTER_LANCZOS4)

    return imgout

def coordinate_shift(bbox, offset):
    cx = bbox[0]
    cy = bbox[1]
    w = bbox[2]
    h = bbox[3]
    sita = bbox[4]


    left = -w/2
    top = -h/2
    right = w/2
    bottom = h/2

    x1, y1 = rotated_coordinate(left, top, sita, cx, cy)
    x3, y3 = rotated_coordinate(right, bottom, sita, cx, cy)

    pt1 = (int(x1+ offset), int(y1+ offset))
    pt3 = (int(x3+ offset), int(y3+ offset))
    return pt1, pt3, sita

def image_processing(white, box,offset, block_size, constance):
    imgout, arc = get_crack(white, box, offset)
    imgout = cv2.cvtColor(imgout, cv2.COLOR_BAYER_BG2GRAY)
    imgout = cv2.medianBlur(imgout, 3)
    imgout = cv2.adaptiveThreshold(imgout, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, constance)
    imgout = 255 -imgout
    return imgout

def crack_split(img, box, roi, offset):
    pt1, pt3, sita = coordinate_shift(box, offset)
    if sita > pi / 2:
        img = rotateImage(img, -degrees(pi - sita), pt1, pt3, roi)
    else:
        img = rotateImage(img, degrees(sita), pt1, pt3, roi)
    return img

def get_dr_pixel(img_path, detected_box_path, block_size, constance):
    img = cv2.imread(img_path, 0)
    bboxes = open(detected_box_path).readlines()
    width = 256
    height = 256
    offset = 300
    black = np.zeros((width + 2 * offset, height + 2 * offset), dtype=np.uint8)
    black[:, :] = 0
    if bboxes != []:
        white = copy.deepcopy(black)
        white[offset:offset + width, offset:offset + height] = img
        cracks = black
        for i in bboxes:
            box = i.split()[1:]
            box = [float(x) for x in box]
            roi = image_processing(white, box, offset, block_size, constance)
            cracks = crack_split(cracks, box, roi, offset)
            crack = cracks[offset:offset + width, offset:offset + height]
        img_add = cv2.addWeighted(img, 0.8, crack, 0.2, 0)
        return img_add, img, crack
    else:
        black = np.zeros((width, height), dtype=np.uint8)
        return black,black,black

def connected_component_analysis(dr, area_th):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dr, connectivity=4)
    output = np.zeros((256, 256), np.uint8)
    list1 = []
    for i in range(1, num_labels):
        if stats[i][-1] < area_th:
            list1.append(i)
    for i, value1 in enumerate(labels):
        for j, value2 in enumerate(value1):
            if value2 in list1:
                labels[i, j] = 0
            if value2 not in list1 and value2 != 0:
                labels[i, j] = 1
                output[i, j] = 255
    return output



### example ##
org_jpg = 'xxxxxxxxxx.jpg'
detected_box_path = 'xxxxxxxxx.txt'
block_size = 21
constance = 5

_,__, dr = get_dr_pixel(org_jpg,detected_box_path,block_size, constance)
dr = 255 - dr
th, dr = cv2.threshold(dr,0,255,cv2.THRESH_OTSU)
dr = 255 - dr
output = connected_component_analysis(dr, 30)





