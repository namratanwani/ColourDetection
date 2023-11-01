# -*- coding: utf-8 -*-

# import required modules
import cv2
import argparse
from scipy.io import loadmat, savemat
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import math
import os

'''
 Please note: To display images plt.imshow() has been used as cv2.imshow() kept on crashing the kernel

 '''

def load_as_double(filename):
    '''
    This function takes file path as argument
    and reads the file path of the image in colored,
    and grayscale format. It returns both format
    '''
    # read image
    img = cv2.imread(filename)
    # convert image to rgb format
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # convert image to gray 
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # convert gray image to float
    img_gray = np.float64(img_gray)
    return img_rgb, img_gray

def convert_image_format_to_uint8(img):
    '''
    This function changes the format of image to unsigned 8 bit interger
    '''
    img_uint8 = cv2.convertScaleAbs(img)
    return img_uint8

def remove_noise(img):
    '''
    This function uses median blur method to remove noise from image.
    The function takes image as argument as 8-bit unsigned integer format
    and returns image with noise removed.
    '''
    img_noiseless = cv2.medianBlur(img,5)
    return img_noiseless

def find_coordinates_of_circles(img, minRadius = 24, maxRadius = 25):
    '''
    This function detects cirlces in the given image of fixed radius. 
    It returns coordinates of cirlces along with its radius.    
    '''
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=3, minDist=120, param1=150, param2=50, minRadius=minRadius, maxRadius=maxRadius)
    circles = np.round(circles[0, :]).astype(int)
    coords = []
    for (x, y, r) in circles:
            # print(x,y)
            cv2.circle(img, (x, y), r, (255, 0, 0), 2)
            # append circle coordinates
            coords.append((x,y))
    return coords,r

def unskew_image(gray_img, color_img, circle_coords):
    '''
    This function performs perspective trnaformation on the the rotated/ skewed image.
    A top-down view of image is returned.
    '''
    
    corner1 = np.array(circle_coords[0])
    corner2 = np.array([circle_coords[2][0],circle_coords[2][1]]) 
    corner3 = np.array(circle_coords[3])
    corner4 = np.array([circle_coords[1][0],circle_coords[1][1]])
    w = 500
    h = 500
    # Define the source and destination points
    src_points = np.array([corner1, corner2, corner3, corner4], dtype=np.float32)
    dst_points = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    # Calculate the perspective transform matrix and apply it to the image
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_rgb = cv2.warpPerspective(color_img, M, (w, h))
    warped_gray = cv2.warpPerspective(gray_img, M, (w, h))
    return warped_gray, warped_rgb

def draw_rectangle(img, coords, r):
    '''
    This fucntion ifinds minimum and maximum values from the second pair of coordinates and uses them to draw rectangle on the image.
    It returns image with rectangle drawn on it.
    '''
    left_min = min(coords[1])
    right_max = max(coords[1])
    # print(left_min, right_max)
    rectangleDrawn = cv2.rectangle(img, (left_min+r, left_min+r), (right_max-r, right_max-r), (0, 0, 255), 2)
    return rectangleDrawn

def crop_image(img, coords, r):
    '''
    This function finds minimum and maximum values from the second pair of coordinates and uses them crop the image.
    It returns a cropped image.
    '''
    left_min = min(coords[1])
    right_max = max(coords[1])
    cropped_image = img[left_min+r: right_max-r, left_min+r: right_max-r]
    return cropped_image

def find_edges(img):
    '''
    This functionn detects edges, uses its contours to detect squares and only adds to the list those coordinates 
    that has area  between desired range. These coordinates are then filtered to remove repetitive values and 
    unnecessary values. Finally, a list of unique square coordinates is returned along with dilated edges.
    '''
    # The values 5 and 75 are threshold values used to determine which pixels are considered edges. 
    # Pixels with gradient values higher than the 75 threshold are considered edges, while pixels 
    # with gradient values lower than the 5 threshold are not considered edges. The edges variable
    # stores the resulting edge map.
    edges = cv2.Canny(img, 5, 75)
    # creates a 2x2 rectangular kernel that is used for dilation, which helps 
    # connect and smooth out the edges.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    # uses the dilation operation to thicken the edges of the image. The dilate function takes the edge
    # map edges, the kernel kernel, and the number of iterations to perform the dilation. The resulting
    # dilated edge map is stored in dilated_edges.
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    # identify the contours (shapes) of the objects in the dilated edge map. 
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    lst = []
    visited = []
    square_coords = []
    for cnt in contours:
        # Approximate contour as a polygon
        approx = cv2.approxPolyDP(cnt, 0.1*cv2.arcLength(cnt, True), True)

        # If polygon has 4 sides, it's a square (in this case)
        if len(approx) == 4:
            # Draw bounding rectangle
            x, y, w, h = cv2.boundingRect(approx)
            area = w * h
            # Check if square is within desired size range
            if 5800 < area < 6600:
                # Retrieve coordinates of square vertices
                square_coords.append((x, y))

    # initialise empty dictionary
    d = {}
    # loop through square coordinates 
    for i in range(0,len(square_coords)-1):
        # get difference between consecutive coordinates
        ans = np.array(square_coords[i]) - np.array(square_coords[i+1])
        # get absolute sum of difference between coordinates
        s = abs(sum(ans))
        # if absolute value is less that 12 then coordinates point to the same square and are close to each other
        if s < 12:
            if i not in d.keys():
                d[i] = square_coords[i]
            if i+1 not in d.keys():
                d[i+1] = square_coords[i]
        # if absolute value is more than 20 then coordinates point to different squares
        elif s > 20:
            if i not in d.keys():
                d[i] = square_coords[i]
            if i+1 not in d.keys():
                d[i+1] = square_coords[i+1]
    # the dictionary still might contain some repitive values
    # this is finally removed by the following code
    final_coords = []
    # iterate through dictionary's values
    for val in d.values():
        # append values if they are not added to final_coords list
        if val not in final_coords:
            final_coords.append(val)

    # print(square_coords)
    #return dilated edges and unique square coordinates
    return dilated_edges, final_coords

def map_coords_to_sections(coords):
    '''
    This function is created to detect which squares are undetected so that
    coordinates can point to their respective squares and can be coorectly 
    aligned in the results.
    '''
    # initialise a dictionary with each key mapping 
    # to row/column number of the square. Here, each key 
    # is a pixel value. 
    
    mapped = {
        23: 0,
        108: 1,
        193: 2,
        278: 3
    }

    # initialise a nested list with tuple (0,0)
    matrix = [[(0,0) for _ in range(4)] for _ in range(4)]
    # iterate through coordinates
    for tup in coords:
        # get x and y coordinates of each set
        x = tup[0]
        y = tup[1]
        # iterate through keys of 'mapped' dictionary
        for key in mapped.keys():
            # find a key that is the closed to x coordinate
            result_x = math.isclose(x, key, abs_tol = 10)
            if result_x:
                # map x coordinate to the closest key's value
                mat_x = mapped[key]
             # find a key that is the closed to y coordinate
            result_y = math.isclose(y, key, abs_tol = 10)
            if result_y:
                # map x coordinate to the closest key's value
                mat_y = mapped[key]
        # print(mat_x, mat_y)
        # print(x,y)
        # each coordinate is now mapped to its respective
        # row and column that determines the location of 
        # its square.
        matrix[mat_y][mat_x] = (x,y)
    # return matrix
    return matrix

def get_color_matrix(img, matrix):

    '''
    This function extracts pixels of the squares determined using the coordinates 
    from the passed matrix. An average value of these pixels for red, green, blue 
    channels help determine the color of the square. A nested list of colours is 
    returned that is of the shape (4,4).
    '''
    # initialise nested list with None values
    colors = [[None for _ in range(4)] for _ in range(4)]
    # iterate over coordinates in the given matrix
    for i in range(4):
        for j in range(4):
            # print(i,j)
            l = matrix[i][j]
            # if coordinate is (0,0)
            if l == (0,0):
                # then the square is undetected
                colors[i][j] = 'undetected'
                # hence, continue iterating next value
                continue
            # extract section of image using given coordinates
            # the width and height of the square is constant
            section = img[l[1]:l[1]+80, l[0]:l[0]+80]
            # get mean of the pixel values in that section
            section_mean = cv2.mean(section)
            # print(section_mean)
            # create a list containing only 1s and 0s 
            # this shows the presence/ absence of rgb values
            binary_list = [1 if val > 130 else 0 for val in section_mean[:3] ]
            # assign rgb values to valriables
            r = binary_list[0]
            g = binary_list[1]
            b = binary_list[2]
            
            # check presence of different channels for different colours
            # and assign color value to each square
            if r and g and b:
                colors[i][j] = 'white'
            elif r and g :
                colors[i][j] = 'yellow'
            elif r and b:
                colors[i][j] = 'purple'
            elif b:
                colors[i][j] = 'blue'
            elif r:
                colors[i][j] = 'red'
            elif g:
                colors[i][j] = 'green'
            else:
                colors[i][j] = 'unknown color'

    # return nested list of colours
    return colors

# get all files to test
files = []
for i in range(1,6):
    files.append('org_'+str(i))
for i in range(1,6):
    files.append('noise_'+str(i))
files

# please update path as per requirement
path = 'masters/CV/images2/images/' 
loc = 1
fig=plt.figure(figsize=(50, 90))
# iterate over files
for file in files:
    # get file path

    filepath = path + file + '.png' 
    # print(filepath)
    # load image 
    rgb_img, gray_img = load_as_double(filepath)
    # convert image to uint8 format
    gray_img_uint8 = convert_image_format_to_uint8(gray_img)
    # remove noise from color image
    rgb_img_noiseless = remove_noise(rgb_img)
    # remove noise from gray image
    gray_img_noiseless = remove_noise(gray_img_uint8)
    # get corner circle coordinates with their radii
    circle_coordinates, radius = find_coordinates_of_circles(gray_img_noiseless)
    # draw rectangle on the image using circle coordinates
    rectangle = draw_rectangle(gray_img_noiseless, circle_coordinates, radius)
    # plt.imshow(rectangle)
    # crop color image using circle coordinates
    cropped_image_rgb = crop_image(rgb_img_noiseless,circle_coordinates, radius)
    # crop gray image using circle coordinates
    cropped_image_gray = crop_image(gray_img_noiseless,circle_coordinates, radius)
    # convert cropped grey image to uint8 format
    cropped_image_gray_img_uint8 = convert_image_format_to_uint8(cropped_image_gray)
    # remove noise from gray image twice for better results
    cropped_image_gray_noiseless_1 = remove_noise(cropped_image_gray_img_uint8)
    cropped_image_gray_noiseless_2 = remove_noise(cropped_image_gray_noiseless_1)
    # get coordinates of squares from the image
    dilated_edges, square_coordinates = find_edges(cropped_image_gray_noiseless_2)
    # map coordinates to respective squares
    mapped_matrix = map_coords_to_sections(square_coordinates)
    # get final results containing strings of colours
    final_matrix = get_color_matrix(cropped_image_rgb, mapped_matrix)
    
    fig.add_subplot(10,1 , loc)
    plt.imshow(cropped_image_rgb)
    plt.title(file+'.png' + ' - '+ str(final_matrix))
    print(file+'.png' + ' - '+ str(final_matrix))
    loc += 1

    # store result in answers directory inside 'path' 
    dirpath = path + 'answers/'
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    savemat(dirpath + file +'.mat', mdict = {'res': final_matrix})
print("Answers stored in .mat format inside 'answers' folder inside", path)

# get all files to test
files_skewed = []
for i in range(1,6):
    files_skewed.append('rot_'+str(i))
for i in range(1,6):
    files_skewed.append('proj_'+str(i))
files_skewed

# The following code is not part of the pipeline and hence, has been coded separately
# to unskew rotated images
path = 'masters/CV/images2/images/'
loc = 1
fig=plt.figure(figsize=(50, 90))
for file in files_skewed:
    # get file path
    filepath = path + file + '.png' 
    # print(filepath)
    rgb_img, gray_img = load_as_double(filepath)
    # convert image to uint8 format
    gray_img_uint8 = convert_image_format_to_uint8(gray_img)
    # remove noise from color image
    rgb_img_noiseless = remove_noise(rgb_img)
    # remove noise from gray image
    gray_img_noiseless = remove_noise(gray_img_uint8)
    # get corner circle coordinates with their radii
    try:
        circle_coordinates, radius = find_coordinates_of_circles(gray_img_noiseless, 24,26)
        undistorted_image_gray, undistorted_image_rgb = unskew_image(gray_img_noiseless, rgb_img_noiseless, circle_coordinates)
        fig.add_subplot(10,1 , loc)
        plt.title(file+'.jpg')
        plt.imshow(undistorted_image_gray)
        loc += 1 
    except:
        pass

