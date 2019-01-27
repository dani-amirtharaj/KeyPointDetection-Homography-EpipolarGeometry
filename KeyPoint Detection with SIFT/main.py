
# Setting up the program and reading input image
import cv2
import numpy as np
image=cv2.imread('Images/img.jpg')
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
imageArray=np.asarray(grayImage)

# Function to generate gaussian kernel given sigma and kernel width/height
def generateGaussianKernel(sigma, oddSize):
    kernelRowIndex = [i for i in range(-(oddSize//2),oddSize//2+1)]
    kernelColIndex = [-1*i for i in range(-(oddSize//2),oddSize//2+1)]
    
    gaussianKernel=[]
    gaussianNormaliser=0
    
    for cIndex in kernelColIndex:
        gaussianKernelCol=[]
        
        for rIndex in kernelRowIndex:
            
            # Equation to generate values for gaussian at different intervals
            gaussianPixel = (1/(2*3.14159*(sigma**2)))*np.exp(
                -1*(((rIndex)**2+(cIndex)**2)/(2*(sigma**2))))
            gaussianNormaliser+= gaussianPixel
            gaussianKernelCol.append(gaussianPixel)
        
        gaussianKernel.append(gaussianKernelCol)
    gaussianKernel=gaussianKernel/gaussianNormaliser
    return gaussianKernel

# Function to compute and add zero pads to the input image, 
# based on the size of the Gaussian kernel
def padImage(imageArray, gaussianKernel):
    
    paddedImage=[]
    imageHpad=[0 for i in range(len(imageArray[0])+(len(gaussianKernel)-1))]
    
    for ind in range(len(gaussianKernel)//2):
        paddedImage.append(imageHpad)
        
    for row in range(len(imageArray)):
        paddedImageRow=[]
        
        for ind in range(len(gaussianKernel)//2):
            paddedImageRow.append(0)
            
        for column in range(len(imageArray[0])):
            paddedImageRow.append(imageArray[row][column])
            
        for ind in range(len(gaussianKernel)//2):
            paddedImageRow.append(0)
        paddedImage.append(paddedImageRow)
        
    for ind in range(len(gaussianKernel)//2):
        paddedImage.append(imageHpad)
    return paddedImage     

# Function to apply gaussian kernel to the input image
def generateFilteredImage (imageArray, gaussianKernel):
    filterResult=[]
    halfKernelSize=len(gaussianKernel)//2
    windowIndex = [i for i in range(-(halfKernelSize),halfKernelSize+1)]
    
    for row in range(halfKernelSize,len(imageArray)-halfKernelSize):
        filterResultRow=[]
        
        for column in range(halfKernelSize,len(imageArray[0])-halfKernelSize):
            pixelResult=0
            
            for rIndex in windowIndex:
                for cIndex in windowIndex:
                    pixelResult+=imageArray[row+rIndex][column+cIndex]*gaussianKernel[
                            rIndex+halfKernelSize][cIndex+halfKernelSize]
            
            filterResultRow.append(pixelResult)
        filterResult.append(filterResultRow)
        
    return filterResult

# Function to generate 4 Octaves for the input image
def generateOctaves(imageArray):
    NUM_OCTAVES = 4;
    octaves=[]
    
    for index in range(NUM_OCTAVES):
        resizedImage=[]
        
        for row in range(0,len(imageArray),2**(index)):
            resizedImageRow=[]
            for column in range(0,len(imageArray[0]),2**(index)):
                resizedImageRow.append(imageArray[row][column])
            resizedImage.append(resizedImageRow)
        
        octaves.append(resizedImage)
    return octaves;

# Function to generate 5 scales for each octave of the input image
def generateScales(octaves):
    
    # Given values of sigma for each octave and scale
    sigmaArray=[[1/(2**0.5),1,2**0.5,2,2*(2**0.5)],
                [2**0.5,2,2*(2**0.5),4,4*(2**0.5)],
                [2*(2**0.5),4,4*(2**0.5),8,8*(2**0.5)],
                [4*(2**0.5),8,8*(2**0.5),16,16*(2**0.5)]]
    scaledOctave=[]
    
    for rIndex in range(len(sigmaArray)):
        scaledOctaveRow=[]
        for cIndex in range(len(sigmaArray[0])):
            
            # Call functions to generate gaussian, add zero pads to image and generate 
            # gaussian blurs for each image in the scale space
            gaussianKernel = generateGaussianKernel(sigmaArray[rIndex][cIndex],7)
            paddedImage = padImage(octaves[rIndex],gaussianKernel)
            filteredImage=generateFilteredImage(paddedImage, gaussianKernel)
            
            scaledOctaveRow.append(filteredImage)
        scaledOctave.append(scaledOctaveRow)
    return scaledOctave            

# Function to get DoG images given the entire scale space
def getDiffGaussian(scaledOctaves):
    diffGaussian=[]
    
    for rIndex in range(len(scaledOctaves)):
        diffGaussianRow=[]
        for cIndex in range(len(scaledOctaves[0])-1):
            diffImage=[]
            
            for row in range(len(scaledOctaves[rIndex][cIndex])):
                diffImageRow=[]
                for column in range(len(scaledOctaves[rIndex][cIndex][0])):
                    
                    diffPixel=scaledOctaves[rIndex][cIndex][row][column
                                ]-scaledOctaves[rIndex][cIndex+1][row][column]
                    diffImageRow.append((diffPixel))  
                    
                diffImage.append(diffImageRow)
            diffGaussianRow.append(diffImage)
            
        diffGaussian.append(diffGaussianRow)
    return diffGaussian;

# Function to detect extrema pixels (maxima and minima), 
# returns pixel locations of maxima and minima
def keyPointDetection(diffGaussian):
    windowIndex=[-1,0,1]
    extremaPixel=[]
    NUM_PIXELS_COMPARED=26
    keyPoints=[]
    
    # Iterate over scale space of 4 octaves
    for rIndex in range(len(diffGaussian)):
        keyPointsOctave=[]
        for cIndex in range(1,len(diffGaussian[0])-1):
            
            keyPointsScale=[]
            indexedImage = diffGaussian[rIndex][cIndex]
            indexedImageAbove = diffGaussian[rIndex][cIndex+1]
            indexedImageBelow = diffGaussian[rIndex][cIndex-1]
            
            for row in range(1,len(indexedImage)-1):
                for column in range(1,len(indexedImage[0])-1):
                    maxPixel=0
                    minPixel=0
                    
                    for rWindowIndex in windowIndex:
                        for cWindowIndex in windowIndex:
                            
                            # Compare value of pixel to neighbours
                            if indexedImage[row][column] > indexedImageAbove[
                                row+rWindowIndex][column+cWindowIndex]:
                                maxPixel+=1
                            if indexedImage[row][column] < indexedImageAbove[
                                row+rWindowIndex][column+cWindowIndex]:
                                minPixel+=1
                            if indexedImage[row][column] > indexedImageBelow[
                                row+rWindowIndex][column+cWindowIndex]:
                                maxPixel+=1
                            if indexedImage[row][column] < indexedImageBelow[
                                row+rWindowIndex][column+cWindowIndex]:
                                minPixel+=1
                            if indexedImage[row][column] > indexedImage[
                                row+rWindowIndex][column+cWindowIndex]:
                                maxPixel+=1
                            if indexedImage[row][column] < indexedImage[
                                row+rWindowIndex][column+cWindowIndex]:
                                minPixel+=1
                                
                            # Break if it cannot be maxima or minima    
                            if maxPixel>0 and minPixel>0:
                                break
                        if maxPixel>0 and minPixel>0:
                                break
                                
                    # Check if pixel is greater than or lesser than 26 neighbours
                    if maxPixel is NUM_PIXELS_COMPARED or minPixel is NUM_PIXELS_COMPARED:
                        extremaPixel=(row,column)
                        keyPointsScale.append(extremaPixel)
             
            # Collect keypoints
            keyPointsOctave.append(keyPointsScale)
        keyPoints.append(keyPointsOctave)
    return keyPoints

# Plot keypoints on original image
def finalKeypointImage(keyPoints):
    keypointsImage=[]
    resizedKeypoints=[]
    for octave in range(len(keyPoints)):
        keyImage=image+0;
        resizedKeypointsRow=[]
        for scale in range(len(keyPoints[0])):
            for i in keyPoints[octave][scale]:
                l=list(i)
                l[0]=l[0]*(2**octave)
                l[1]=l[1]*(2**octave)
                keyImage[tuple(l)]=255;
                resizedKeypointsRow.append(tuple(l))
        resizedKeypoints.append(resizedKeypointsRow)
        keypointsImage.append(keyImage)
    return keypointsImage, resizedKeypoints

# Call funtions to generate octaves, scale space, DoG and find keypoints
octaves=generateOctaves(imageArray)
scaledOctaves=generateScales(octaves)
diffGaussian=getDiffGaussian(scaledOctaves)
keyPoints=keyPointDetection(diffGaussian)
keypointsImage, resizedKeypoints=finalKeypointImage(keyPoints)

# Used in getting the left most keypoints in the image
def compare(l):
    return l[1]

# Save keypoints for each octave overlayed on the original image
for i in range(len(keypointsImage)):
        cv2.imshow('Keypoints',np.asarray(keypointsImage[i]))
        cv2.imwrite('Results/Keypoints_Octave_'+str(i+1)+'.jpg',np.asarray(keypointsImage[i]))
        cv2.waitKey(0)
cv2.destroyAllWindows()

# Print the Left most points in an image
for i in range(len(resizedKeypoints)):
    print("\n Left Most Keypoints in octave "+str(i+1)+": \n")
    resizedKeypoints[i].sort(key=compare)
    for j in range(10):
        try:
            print(resizedKeypoints[i][j])
        except:
            None

# Plot keypoints on black image, for better visualization
blackSize=(len(imageArray),len(imageArray[0]))
for octave in range(len(keyPoints)):
    black=np.zeros(blackSize);
    for scale in range(len(keyPoints[0])):
        for i in keyPoints[octave][scale]:
            l=list(i)
            l[0]=l[0]*(2**octave)
            l[1]=l[1]*(2**octave)
            black[tuple(l)]=255;
        cv2.imshow('BlackKeyPoints',np.asarray(black))
        cv2.imwrite('Results/BlackKeypoints_Octave_'+str(octave+1)+'.jpg',np.asarray(black))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Display Scale Space
for i in range(len(scaledOctaves)):
    for j in range(len(scaledOctaves[0])):
        cv2.imshow('Scale Space',np.asarray(scaledOctaves[i][j])/255)
        cv2.imwrite('Results/Octave_'+str(i+1)+'_Scale_'+str(j+1)+
                    '.jpg',np.asarray(scaledOctaves[i][j]))
        cv2.waitKey(0)
cv2.destroyAllWindows()

# Display DoG images
for i in range(len(diffGaussian)):
    for j in range(len(diffGaussian[0])):
        cv2.imshow('DoG',np.asarray(diffGaussian[i][j]))
        cv2.imwrite('Results/DoG_Octave_'+str(i+1)+'_Scale_'+str(j+1)+
                    '.jpg',np.asarray(diffGaussian[i][j])*255)
        cv2.waitKey(0)
cv2.destroyAllWindows()