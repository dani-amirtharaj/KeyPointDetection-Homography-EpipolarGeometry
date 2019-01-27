
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


image1=cv2.imread('Images/tsucuba_left.png')
image2=cv2.imread('Images/tsucuba_right.png')

# Detecting keypoints and computing keypoint descriptors for the 2 inout images
keypointsImage1, descriptorImage1 = applySIFT(image1)
keypointsImage2, descriptorImage2 = applySIFT(image2)

# Writing the matches detected in the 2 images to the filesystem
Image1Keypoints=cv2.drawKeypoints(image1,keypointsImage1,None)
cv2.imwrite('Results/task2_sift1.jpg',Image1Keypoints)
Image2Keypoints=cv2.drawKeypoints(image2,keypointsImage2,None)
cv2.imwrite('Results/task2_sift2.jpg',Image2Keypoints)

# Get good matches using KNN algorithm between kepoint descriptors of 
# 2 input images
goodList, good = getGoodMatches(descriptorImage1, descriptorImage2)

# Plotting knn matches based on the keypoint distances computed
# cv2.drawMatchesKnn expects list of lists as matkeypointsches
imagePlot = cv2.drawMatchesKnn(image1,keypointsImage1,image2,keypointsImage2,goodList,None,flags=2)
cv2.imwrite('Results/task2_matches_knn.jpg',imagePlot)

# Getting keypoint locations as an array of (x,y) pixel coordinates
ptsImage1 = np.int32(np.round([keypointsImage1[m.queryIdx].pt for m in good]).reshape(-1,1,2))
ptsImage2 = np.int32(np.round([keypointsImage2[m.trainIdx].pt for m in good]).reshape(-1,1,2))

# Getting Fundamental matrix after applying RANSAC on
# well matched keypoints on both images 
F, mask = cv2.findFundamentalMat(ptsImage1,ptsImage2,cv2.RANSAC,1)

# Get 10 inlier matches after applying RANSAC
matchesMask = getInliers(mask, 10)
inlierImage = cv2.drawMatches(image1,keypointsImage1,image2,keypointsImage2,
                              good,None,matchesMask = matchesMask,flags = 2)

# Applying mask on the keypoints to get 10 inlier pairs
ptsImage1 = ptsImage1[np.array(matchesMask).ravel() == 1]
ptsImage2 = ptsImage2[np.array(matchesMask).ravel() == 1]


# Iterate over 10 inliers to calculate epilines
h ,w, d = image1.shape
for i in range(len(ptsImage1)):
    
    color = tuple(np.random.randint(0,255,3).tolist())
    
    # Calculate epilines on image 2 for keypoints on image1
    line2 = cv2.computeCorrespondEpilines(ptsImage1[i], 1, F)
    
    # Calculate epilines on image 1 for keypoints on image2
    line1 = cv2.computeCorrespondEpilines(ptsImage2[i], 2, F)
    
    # Compute 2 sample points on each line for plotting
    p1 = map(int , [0,-line1.ravel()[2]/line1.ravel()[1]])
    p2 = map(int, [w, -(line1.ravel()[2]+line1.ravel()[0]*w)/line1.ravel()[1]])
    p3 = map(int , [0,-line2.ravel()[2]/line2.ravel()[1]])
    p4 = map(int, [w, -(line2.ravel()[2]+line2.ravel()[0]*w)/line2.ravel()[1]])

    # Draw epilines on corresponding image
    image1EpipolarLines = cv2.line(image1, tuple(p1), tuple(p2), color,1)
    image2EpipolarLines = cv2.line(image2, tuple(p3), tuple(p4), color,1)
    
    # Draw keypoints on corresponding image
    image1Final = cv2.circle(image1EpipolarLines, tuple(ptsImage1[i].ravel()), 2, color,5)
    image2Final = cv2.circle(image2EpipolarLines, tuple(ptsImage2[i].ravel()), 2, color,5)

print('Fundamental Matrix:')
print(F)

cv2.imwrite('Results/task2_epi_right.jpg',image2Final)
cv2.imwrite('Results/task2_epi_left.jpg',image1Final)

# Compute disparity between 2 images to create depth map
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=21)
disparity = stereo.compute(cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY),cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY))

plt.imsave('Results/task2_disparity.jpg',disparity, cmap='gray')
