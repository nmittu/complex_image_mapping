import sys
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from numpy import nan, isnan
import time
import cv2
from PIL import Image

def tuple_mean(t1, t2):
    t = []
    for i in range(len(t1)):
        t.append(int((t1[i] + t2[i])/2))

    return tuple(t)

def blend(img1, img2):
    whiteThresh = 230

    img = Image.new('RGBA', img1.size)

    width, height = img.size

    pdata1 = img1.load()
    pdata2 = img2.load()
    pdata = img.load()

    for y in range(height):
        for x in range(width):
            if np.mean(pdata1[x, y][:-1]) > whiteThresh:
                pdata[x, y] = pdata2[x, y]
            elif np.mean(pdata2[x, y][:-1]) > whiteThresh:
                pdata[x, y] = pdata1[x, y]
            else:
                pdata[x, y] = tuple_mean(pdata1[x, y], pdata2[x, y])

    return img


def correct_img(img, outputResolution, mappedImage, mappedImageInt):
    mask = np.zeros((outputResolution[0], outputResolution[1]), dtype='uint8')
    mask[isnan(mappedImage[:, :, 0])] = 1

    maskCentral = mask.copy()[300:900, 300:900]
    mappedImageCentral = mappedImageInt.copy()[300:900, 300:900,:]

    startTime = time.time()
    finalImage = cv2.inpaint(mappedImageCentral, maskCentral, 3, cv2.INPAINT_NS)
    print('Time Elapsed = ' + str(time.time() - startTime) + 's.')
    return finalImage

def map_half(img, inputBounds):
    inputResolution = img.shape

    outputBounds = 6.5*np.array([-1,1,-1,1])
    outputResolution = [1200,1200]

    xInput=np.linspace(inputBounds[0], inputBounds[1], inputResolution[1])
    yInput=np.linspace(inputBounds[2], inputBounds[3], inputResolution[0])

    x,y=np.meshgrid(xInput,yInput)
    z=x+y*1j

    w = z**2

    reSlope = float(outputResolution[1])/(outputBounds[1]-outputBounds[0])
    imSlope = float(outputResolution[0])/(outputBounds[3]-outputBounds[2])

    reIndex = (w.real*reSlope + outputResolution[1]/2).round().astype('int')
    imIndex = (w.imag*imSlope + outputResolution[0]/2).round().astype('int')

    outputImageSize = outputResolution[:]
    outputImageSize.append(3)

    mappedImage = np.empty(outputImageSize)
    mappedImage[:] = nan

    whiteThresh = 230
    startTime = time.time()

    for i in range(inputResolution[0]):
        for j in range(inputResolution[1]):
            #Only map pixels in the output range:
            if reIndex[i, j] > 0 and reIndex[i, j] < outputResolution[1]:
                if imIndex[i,j] > 0 and imIndex[i,j] < outputResolution[0]:
                    #Check if that pixel has been filled yet:
                    if isnan(mappedImage[imIndex[i,j], reIndex[i,j], 0]):
                        mappedImage[imIndex[i,j], reIndex[i,j], :] =  img[i, j, :]
                    #Check if non-white:
                    elif img[i, j, 0]< whiteThresh or img[i, j, 1]< whiteThresh or \
                        img[i, j, 2:]< whiteThresh:
                        mappedImage[imIndex[i,j], reIndex[i,j], :] =  img[i, j, :]

    print('Time Elapsed = ' + str(time.time()-startTime) + 's.')

    mappedImageInt = mappedImage.astype('uint8')

    return correct_img(mappedImageInt, outputResolution, mappedImage, mappedImageInt)

def map(img):
    imgr = map_half(img[:,int(img.shape[1]/2):], np.array([0, 2, -2, 2]))
    imgl = map_half(img[:,:int(img.shape[1]/2)], np.array([-2, 0, -2, 2]))
    return blend(Image.fromarray(imgr), Image.fromarray(imgl))

img = map(mpimg.imread(sys.argv[1]))

plt.imshow(img)
plt.show()
