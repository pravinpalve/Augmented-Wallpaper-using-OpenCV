import cv2 #for doing image processing operations
import matplotlib.pyplot as plt #for plotting results
import numpy as np #for data processing operations
import cv2.aruco as aruco   #for detecting aruco markers

##### Read images #####
img_list= [1,2,3,4,5,6,7,8,9,10,11]  #list of image names
for img_num in img_list:  #for looping through all images in one go
#img_num=1   #use this line if you want to process only certain one image, don't forget to comment out above one.
img_path= f"C:/Users/pravi/OneDrive/Documents/Python/COMPUTER VISION/{img_num}.jpg" #this path will differ from system to system, it's where images from source image folder would go
display_img_path  = r"C:\Users\pravi\OneDrive\Documents\Python\COMPUTER VISION\CV TASK 1 DISPLAY IMAGE.jpg"  # Insert display image path here. r"" is used for specifying a raw string literal in Python. It prevents backslashes (\) from being interpreted as escape sequences
image = cv2.imread(img_path)    #The cv2.imread function is used to read an image from the file system.
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #converting color format
display_image = cv2.imread(display_img_path)
display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)

##### detecting and reading Aruco markers
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250) #dictionary containing 250 unique black and white markers arranged in a 6x6 grid pattern.
parameters = aruco.DetectorParameters()
corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters) #cv2.aruco.detectMarkers function is used to identify and localize Aruco markers within the image
print(corners)

##### Only corner points are useful (getting them in readable format)
corner = np.reshape(corners,(-1)) #corners is 2D matrix eg. [[x,y],[x,y],[x,y],[x,y]]. this command flattens it out to [x,y,x,y,x,y,x,y]
corner = corner.astype(int) #entries of the [corners] are in float format. this converts them into integers 

##### Defining original image points and perspective transfromation points
(x, y) = (155,95) # Adjust display_image scaling here ) here we make an extra rectangle on the image to be displayed. this rectangle will be smaller than the actual image. this rectangle will be mapped on the aruco marker corners. hence smaller the rectangle larger the image. x and y values are basically margins between actual image and the rectangle. 
(tr, tl, br, bl) = ([0+x, 0+y], [display_image.shape[1]-x, 0+y], [display_image.shape[1]-x, display_image.shape[0]-y], [0+x, display_image.shape[0]-y])
display_img_corner_pts = np.float32([tr, tl, br, bl]) # Four corner coordinates of the display image (to be honest, these are the four corners ofadded rectangle.)
perspective_shifted_pts = np.float32([[corner[0], corner[1]], [corner[2], corner[3]], [corner[4], corner[5]], [corner[6], corner[7]]])

#################### Perspective transform of display image #######################

# This is a transfirmation matrix to transform our image into different perspective
matrix = cv2.getPerspectiveTransform(display_img_corner_pts, perspective_shifted_pts) #it calculates a transformation matrix which tranfer points in first argument (src-source) to points in second argument (dst-destination)
#print(matrix) #just to see the generated matrix. not essential for working of the code. you can comment it out.

# here matrix generated in previous step will be actually used to wrap the source image to shape we got from perspective. 
transformed_image = cv2.warpPerspective(display_image, matrix, (image.shape[1], image.shape[0])) 

###################### New mask #################

b, g, r = cv2.split(transformed_image) #separates a color image into its individual color channels

mask_r = np.zeros_like(r) # This line creates a new NumPy array named mask_r with the same shape and data type as the r array (which holds the red channel values).
mask_r[r != 0] = 255  #every entry which is not 0 will be set to 255. 
mask_g = np.zeros_like(g)
mask_g[g != 0] = 255
mask_b = np.zeros_like(b)
mask_b[b != 0] = 255 

mask = cv2.merge((mask_b, mask_g, mask_r))
########################################################

inverted_mask = np.ones_like(mask) * 255 - mask  #this line inverts the image

########################################################

# Bitwise_and retains pixel values in original image where the mask values are nonzero and fills rest with (0,0,0)
masked_image1 = cv2.bitwise_and(image, inverted_mask)  #all the white pixes will be replaced with pixels from original "image"

########################################################

# Pixel values are added fromt the two images
add_image =  cv2.add(masked_image1, transformed_image)

########################################################

#plt.imshow(transformed_image) 
#plt.imshow(mask) #need
#plt.imshow(inverted_mask) 
#plt.imshow(masked_image1) 
plt.imshow(add_image) 
plt.show()