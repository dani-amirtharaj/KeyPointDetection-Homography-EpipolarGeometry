
# Setting up program
import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse

# Setting seed for reproducibility
UBIT = 'damirtha'
np.random.seed(sum([ord(c) for c in UBIT]))


# Function to apply SIFT to given image
def applySIFT(image):
    
    # Creating SIFT object, which will be used for applying SIFT on images
    sift = cv2.xfeatures2d.SIFT_create()

    # Detecting keypoints and computing keypoint descriptors for the inout images
    keypointsImage, descriptorImage = sift.detectAndCompute(image,None)
    
    return keypointsImage, descriptorImage

# Function to get good matches, given feature descriptors
def getGoodMatches(descriptorImage1, descriptorImage2):
    
    # Applying Brute Force matcher for getting K nearest neighbours for 
    # each keypoint using respective descriptors
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptorImage1, descriptorImage2, k=2)

    goodList = []
    good = []

    # Filtering good matches based on a distance of 0.75 between 
    # keypoint pairs in 2 images
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            goodList.append([m])
            good.append(m)
            
    return goodList, good

# Function to get Inliers after RANSAC, accepts mask returned
def getInliers(mask, num=10):
    matchesMask = mask.ravel().tolist()
    indices = []
    for ind in range(len(matchesMask)):
        if matchesMask[ind] == 1:
            indices.append(ind)
    matchesMask = [0]*len(matchesMask)
    np.random.shuffle(indices)
    indices = indices[:num]
    for ind in indices:
            matchesMask[ind] = 1
    return matchesMask


# Function to get left most and top most edge points, to translate warped image
def getExtremePoints(image1CornersPlane2):
    
    # Computing the left most point (point corr. to xmin)
    xMin = min(image1CornersPlane2[0][0], image1CornersPlane2[1][0])

    # Computing the top most point (point corr. to ymin)
    yMin = min(image1CornersPlane2[0][1], image1CornersPlane2[3][1])

    # Computing the right most point (point corr. to xmax)
    xMax = max(image1CornersPlane2[2][0], image1CornersPlane2[3][0])

    # Computing the lowest point (point corr. to ymax)
    yMax = max(image1CornersPlane2[1][1], image1CornersPlane2[2][1])
    return xMin, yMin, xMax, yMax


# Reading images
image1=cv2.imread('Images/mountain1.jpg')
image2=cv2.imread('Images/mountain2.jpg')

# Detecting keypoints and computing keypoint descriptors for the 2 inout images
keypointsImage1, descriptorImage1 = applySIFT(image1)
keypointsImage2, descriptorImage2 = applySIFT(image2)

# Writing the matches detected in the 2 images to the filesystem
Image1Keypoints=cv2.drawKeypoints(image1,keypointsImage1,None)
cv2.imwrite('Results/task1_sift1.jpg',Image1Keypoints)
Image2Keypoints=cv2.drawKeypoints(image2,keypointsImage2,None)
cv2.imwrite('Results/task1_sift2.jpg',Image2Keypoints)

# Get good matches using KNN algorithm between kepoint descriptors of 
# 2 input images
goodList, good = getGoodMatches(descriptorImage1, descriptorImage2)

# Plotting knn matches based on the keypoint distances computed
# cv2.drawMatchesKnn expects list of lists as matkeypointsches
imagePlot = cv2.drawMatchesKnn(image1,keypointsImage1,image2,keypointsImage2,goodList,None,flags=2)
cv2.imwrite('Results/task1_matches_knn.jpg',imagePlot)

# Getting keypoint locations as an array of (x,y) coordinates
ptsImage1 = np.array([ keypointsImage1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
ptsImage2 = np.array([ keypointsImage2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

# Getting homography matrix after applying RANSAC on
# well matched keypoints on both images with projection error <= 1
H, mask = cv2.findHomography(ptsImage1, ptsImage2, cv2.RANSAC)
print('Homography Matrix:')
print(H)
 
# Get 10 inlier matches after applying RANSAC
matchesMask = getInliers(mask, 10)
inlierImage = cv2.drawMatches(image1,keypointsImage1,image2,keypointsImage2,
                              good,None,matchesMask = matchesMask,flags = 2)
cv2.imwrite('Results/task1_matches.jpg',inlierImage)

# Getting corners of image 1 in the 2nd plane
h, w, d = image1.shape
image1Corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
image1CornersPlane2 = np.squeeze(cv2.perspectiveTransform(image1Corners,H))

# The following function gives the max dimensions that image1 can
# take in the second plane (for displaying all pixels)
xMin, yMin, xMax, yMax = getExtremePoints(image1CornersPlane2)

# Using the max dimensions of image1 and combinations of 
# image 1 and 2 to find shape of output image 
t1 = (xMax-xMin, yMax-yMin)
t2 = (len(image2[0])-int(xMin), len(image2)-int(yMin))
finalImageShape = max(t1,t2)

# Translation matrix, to translate image1 by dimensions which go out 
# of the image returned from perspective warp due to -ve locations
if xMin < 0 and yMin < 0:
    translate = np.float32([[1,0, -xMin], [0,1, -yMin], [0,0,1]])
elif xMin < 0:
    translate = np.float32([[1,0, -xMin], [0,1,0], [0,0,1]])
elif xMin < 0:
    translate = np.float32([[1,0,0], [0,1, -yMin], [0,0,1]])
else:
    translate = np.float32([[1,0,0], [0,1,0], [0,0,1]])

# Applying homography to image1 to warp it to image2 with translation
finalImage = cv2.warpPerspective(image1, np.matmul(translate,H), finalImageShape)

# Slicing the image with warped image1 to place image2 as well
finalImage[-int(yMin):-int(yMin)+len(image2), -int(xMin):-int(xMin)+len(image2[0])]=image2
cv2.imwrite('Results/task1_pano.jpg',finalImage)

