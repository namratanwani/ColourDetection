# Colour Detection

## Introduction
In image processing, colour detection refers to the process of identifying and extracting colours from an image. The process involves analysing pixels of an image and determining their colour in terms of RGB (red, green, blue) values or other colour spaces such as HSV (hue, saturation, value). It has several applications such as: object recognition, image segmentation, colour-based tracking, etc. There are numerous ways to detect and determine the colours of an image using different techniques. Our approach is performed in Python and uses OpenCV module and is defined below.

## Approach
1.	load_as_double(filepath)
This function takes in the file path of the image and reads it in BGR format which is the default reading format. It changes the format of image to RGB and gray. It’s important to store colour information of the image to detect colours and hence it is stored in RGB format. Also, RGB format is one of the most common formats used. This function also changes the format of loaded image to gray so that other functions in OpenCV library can process image effortlessly. The gray image is then converted into float data type. This is done to include double precision in the image that provides a greater range of values and more accurate calculations. The function returns the image in RGB format and double of grey image.

2.	convert_image_format_to_uint8(gray_img)
Some functions in OpenCV process images when the values of pixels are unsigned 8-bit integers. Hence, this function simply converts images to 'uint8' format and returns them.

3.	remove_noise(rgb_img)
Noise hampers the quality of an image. It is important to remove or reduce noise as much as possible to analyse, diagnose and extract insights from images in a better way. This process is aimed at improving the accuracy and reliability of results.
We use median blur to reduce noise as median blur works better with salt and pepper noise than gaussian blur. It simply replaces each pixel with the median value of its neighbouring pixels. The method takes in two parameters: first is he image, and second is the kernel size. The more is the size of kernel the more will be the blur effect.

4.	find_coordinates_of_circles(gray_img_noiseless)
To detect corner circles of the image, this function uses Hough transform method as it is less affected by noise and can detect circles regardless of their size, location or orientation of the image. Specifically for circles, it is called HoughCircles() and it takes several parameters. The first parameter is the image, the second parameter is the method used to detect circles. The third dp parameter is a ratio that with smaller values will give better results but will take more time to process. minDist is the minimum distance between two detected circles. param1 is the threshold for edge detection. param2 is the threshold for circle detection. minRadius and maxRadius are the minimum and maximum radii of the detected circles.

5.	unskew_image(gray_img, color_img)
This function takes in gray and colour images. The aim is to find corners of the skewed object in the image as source points and use width and height of object to determine destination points. With these points a perspective transformation is performed and a top-down view of the image is returned.

6.	draw_rectangle(gray_img_noiseless, circle_coordinates, radius)
This function simply draws a rectangle on the image using coordinates of the detected circle and returns it.

7.	crop_image(rgb_img_noiseless,circle_coordinates, radius)
This function crops image using the coordinates of the circle detected. This is done to get the part of image that is of our interest.

8.	find_edges(cropped_image_gray_noiseless)
The function first performs edge detection using Canny() method, dilates the edges using a pre-defined kernel to connect and smoothen them and find contours of the objects using findContours() method. The first for loop checks if the contour is of a square and then checks if its area lies in the required range. The last for loop filters and store unrepetitive coordinates of squares. The function finally returns dilated edges and the coordinates of all detected squares.

9.	map_coords_to_sections(square_coordinates)
This function simply maps the square coordinates to their respective positions in result matrix. This is important to display results in correct format. It returns a matrix with mapped values.

10.	get_color_matrix(cropped_image_rgb, mapped_matrix)
This function simply uses the colour information of image to find mean of RGB values of pixels of squares. These mean values are used to detect presence of certain colours. It then returns a nested list of size (4,4) containing colour strings.


## Results
| Filename | Output | Success | Notes |
| --- | --- | --- | --- |
| org\_1.png | [['yellow', 'white', 'blue', 'red'], ['white', 'green', 'yellow', 'white'], ['green', 'blue', 'red', 'red'],['yellow', 'yellow', 'yellow', 'blue']] | Yes | Detected 16/16 squares correctly |
| org\_2.png | [['blue', 'yellow', 'blue', 'blue'], ['white', 'red', 'white', 'yellow'], ['green', 'yellow', 'green', 'yellow'], ['yellow', 'blue', 'green', 'red']] | Yes | Detected 16/16 squares correctly |
| org\_3.png | [['green', 'yellow', 'red', 'blue'], ['blue', 'yellow', 'blue', 'blue'], ['white', 'blue', 'green', 'green'], ['white', 'blue', 'blue', 'yellow']] | Yes | Detected 16/16 squares correctly |
| org\_4.png | [['green', 'yellow', 'blue', 'white'], ['red', 'blue', 'white', 'white'],['green', 'yellow', 'yellow', 'blue'], ['blue', 'blue', 'blue', 'white']] | Yes | Detected 16/16 squares correctly |
| org\_5.png | [['yellow', 'purple', 'red', 'green'], ['red', 'purple', 'green', 'red'],['yellow', 'yellow', 'red', 'white'], ['yellow', 'white', 'green', 'red']] | Yes | Detected 16/16 squares correctly |
| noise\_1.png | [['undetected', 'white', 'red', 'undetected'], ['white', 'undetected', 'red', 'undetected'], ['undetected', 'red', 'undetected', 'undetected'], ['green', 'yellow', 'green', 'green']] | No | Detected 9/16 squares correctly |
| noise\_2.png | [['green', 'yellow', 'undetected', 'blue'], ['undetected', 'white', 'undetected', 'undetected'], ['red', 'undetected', 'red', 'undetected'], ['yellow', 'undetected', 'red', 'white']] | No | Detected 9/16 squares correctly |
| noise\_3.png | [['green', 'undetected', 'red', 'red'], ['green', 'red', 'undetected', 'red'], ['purple', 'red', 'red', 'white'],['purple', 'red', 'white', 'white']] | No | Detected 14/16 squares correctly |
| noise\_4.png | [['green', 'yellow', 'green', 'green'], ['white', 'yellow', 'red', 'red'],['blue', 'yellow', 'blue', 'blue'],['white', 'blue', 'red', 'white']] | Yes | Detected 16/16 squares correctly |
| noise\_5.png | [['white', 'red', 'white', 'yellow'],['red', 'yellow', 'white', 'yellow'], ['green', 'red', 'green', 'red'],['red', 'yellow', 'purple', 'white']] | Yes | Detected 16/16 squares correctly |

From the above results, we can see that 7 out of 10 images are detected successfully. Therefore, the success rate of the algorithm is 70%, given that a success is when the algorithm detects all squares correctly. Also, going by the number of squares that are detected correctly, the detection rate of the algorithm is 90%.

## Evaluation & Analysis
The algorithm fails to detect some squares when images are too noisy. To solve this problem, we can create a general solution for all images that extracts coordinates of squares from the ideal (no noise) image and uses those coordinates to detect squares in other images. This way noisy images won’t be depended on their contours. 
For real photos, we can take advantage of the white background and use the objects with white background only. This can be done by thresholding the image. Also, real images will require a lot of noise removal which can be done by Gaussian blur method. It will also require undistorting the skewed images. Some images are quite already blurred so they may also need to be sharpened. Each image is quite different from each other, hence would require human intervention for appropriate results.

## Conclusion
In conclusion, we have performed thorough analysis with code in Python to detect colours from the images provided.
