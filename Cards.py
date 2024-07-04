import tensorflow as tf
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#**********Images necessary for it to work**************************************************
dark  = cv.imread('Pictures/Attributes/Dark.png')
light = cv.imread('Pictures/Attributes/light.png')
water = cv.imread('Pictures/Attributes/water.png')
earth = cv.imread('Pictures/Attributes/earth.png')
fire  = cv.imread('Pictures/Attributes/fire.png')
wind  = cv.imread('Pictures/Attributes/wind.png')
spell = cv.imread('Pictures/Attributes/spell.png')
trap  = cv.imread('Pictures/Attributes/trap.png')

#alpha = 1.5 # Contrast control
#beta = 10 # Brightness control

spell = cv.convertScaleAbs(spell, alpha=0.5, beta=11)
water = cv.convertScaleAbs(water, alpha=0.7, beta=10)
wind  = cv.convertScaleAbs(wind, alpha=0.5, beta=10)

array = []
array.append(dark)
array.append(light)
array.append(water)
array.append(earth)
array.append(fire)
array.append(wind)
array.append(spell)
array.append(trap)
#*************************************************************************************************  

def getlevel(img):
    cropped = img[140:200, 40:737]
    cv.imshow('zoom',cropped)
    gray = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
    gray = cv.bilateralFilter(gray,5,150,300)
    cv.imshow('gray',gray)


    ret,thresh1 = cv.threshold(gray,127,255,cv.THRESH_BINARY)

    plt.imshow(thresh1,'gray',vmin=0,vmax=255)
    print("Status check All clear")


    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 15, param1 = 5, param2 = 17, minRadius = 25, maxRadius = 30) 

    print(circles)
    circle = np.uint16(np.around(circles))
    canny = cv.Canny(gray, 5, 100)

    cv.imshow('canny',canny)
    cv.waitKey(0)

    for (x,y,r) in circle[0,:]:
        cv.circle(cropped, (x,y),r,(0,255,0),3)
        cv.circle(cropped, (x,y),2,(0,255,255), 3)

    cv.imshow('img', cropped)
    cv.waitKey(0)
    level = len(circles[0])
    print("\n\nThis Monster is a Level ", level, " Card")
    return level
#*******************************************************************************************************************************

def getattribute(img):
    
    stringarray = ["dark", "light","water","earth", "fire", "wind", "spell", "trap"]
   
    maskcircle = cv.imread('Pictures/SampleCards/mask2.jpg') #MASK
    maskcircle = cv.cvtColor(maskcircle, cv.COLOR_BGR2GRAY)
    maskcircle = cv.resize(maskcircle, (75,77))
    mask1,maskthresh = cv.threshold(maskcircle,127,255,cv.THRESH_BINARY)

    gray = array
    ret = gray
    thresh = gray
    
    for i in range(8):
        gray[i] = cv.cvtColor(array[i], cv.COLOR_BGR2GRAY)
        gray[i] = cv.resize(gray[i], (75,77))
        ret[i],thresh[i] = cv.threshold(gray[i], 127, 255, cv.THRESH_BINARY)  #change to THRESH_TRUNC FOR SOME LIGHT COLORED IMAGES SUCH AS WIND



    #Cropping out just the 'attribute' portion of the image

    cropped = img[52:129, 680:755]
    cropped = cv.convertScaleAbs(cropped, alpha=0.55, beta=10)
    cropped = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
    cropped = cv.bitwise_or(cropped,maskthresh)

    cv.imshow('gray', cropped)
    cv.waitKey(0)

    ret,thresh1 = cv.threshold(cropped,127,255,cv.THRESH_BINARY)

    inputpixels = cv.countNonZero(thresh1) # # of white pixels in input image
    result = array
    for i in range(8):
        result[i] = cv.bitwise_and(thresh1,thresh[i])


    pixelcnt = []
    for i in range(8):
        pixelcnt.append(cv.countNonZero(result[i]))

    plt.subplot(2,3,2)
    plt.title("AND DARK")
    plt.imshow(result[0],'gray',vmin=0,vmax=255)

    largest = max(pixelcnt)
    for i in range(8):
        if (pixelcnt[i] == largest):
            index = i
            break
    print("This card type is a ", stringarray[index], " attribute")
    return stringarray[index]


def getType(img):
    colorarray = ["Ritual", "Effect", "Fusion", "Normal", "Synchro", "Xyz"]
    mask = cv.imread('Pictures/SampleCards/CardMask.jpg')
    mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    ret3,maskthresh = cv.threshold(mask,127,255,cv.THRESH_BINARY)

    plt.subplot(2,3,1)
    plt.imshow(maskthresh,'gray',vmin=0,vmax=255)
    plt.subplot(2,3,2)
    plt.imshow(img,'gray',vmin=0,vmax=255)

    #Cover the card image and saving it into a new image variable
    img1a = cv.bitwise_and(img,img, mask=maskthresh)
    cv.imshow('maskedimage',img1a)
    cv.waitKey(0)

#Cropping out the Borders since the original size is 813x1185
    img1a = img1a[26:1155, 26:783]

    HSVimg = cv.cvtColor(img1a, cv.COLOR_BGR2HSV)
    cv.imshow('HSVimg',HSVimg)
    

#B64B05 average effect card color [182,75,5] RGB
#Defining the range

#**************COLOR BOUNDARIES IN HSV FOR THE DIFFERENT COLORED CARDS THAT ARE AVAILABLE*************************************

    #Works for Blue Detects Ritual Monsters
    effect_lower = np.array([100,50,50], np.uint8) 
    effect_upper = np.array([130,255,255], np.uint8) #HSV notation 

    #works for Brown Detects Effect Monsters  RGB: Darkest Brown: [180,88,41] Lightest Brown: [230,202,190] -> [9  44 230]
    #[[[ 10 197 180]]]
    effect_lower1 = np.array([0,50,50], np.uint8) 
    effect_upper1 = np.array([10,255,255], np.uint8) #HSV notation 

    #works for fusion Monsters: RGB: Dark Purple:[130,55,46] -> [3 165 130], Light Purple: [222,199,228] -> [144  32 228]
    effect_lower2 = np.array([140,50,50], np.uint8) 
    effect_upper2 = np.array([160,255,255], np.uint8) #HSV notation 

    #works for normal monsters: RGB Dark Yellow:[197,149,75] ->  [18 158 197] , Light Yellow: [236,222,196]->[ 20  43 236]
    effect_lower3 = np.array([18,50,50], np.uint8) 
    effect_upper3 = np.array([20,255,255], np.uint8) #HSV notation 

    #works for synchro monsters: RGB 'Dark White':[222, 221, 219] -> [ 20   3 222] , 'Light White':[245, 244, 242] ->  [20   3 245]
    effect_lower4 = np.array([15,3,200], np.uint8) 
    effect_upper4 = np.array([20,14,255], np.uint8) #HSV notation 

    #works for XYZ monsters: RGB 'Dark Black':[0,0,0] -> [0,0,0]  'Light Black:' [39, 38, 36]    ->  [20 20 39]
    #rgb(34, 34, 36) -> [120  14  36]       
    effect_lower5 = np.array([0,0,12], np.uint8) 
    effect_upper5 = np.array([20,20,39], np.uint8) #HSV notation 
#********************************************************************************************************************

    effectsmatrixlower = [effect_lower, effect_lower1, effect_lower2, effect_lower3, effect_lower4, effect_lower5]
    effectsmatrixupper = [effect_upper, effect_upper1, effect_upper2, effect_upper3, effect_upper4, effect_upper5]

    kernel = np.ones((5, 5), "uint8") 
    #Creating the Mask based off the boundaries
    i = 0
    whitepixels = 0
    while(whitepixels < 200000):
        low = effectsmatrixlower[i]
        high = effectsmatrixupper[i]

        effect_mask = cv.inRange(HSVimg, low, high) 
        effect_mask = cv.dilate(effect_mask, kernel) 
        res_effect = cv.bitwise_and(HSVimg, HSVimg, mask = effect_mask) 

        whitepixels = cv.countNonZero(effect_mask)  
        print("WhitePixels: ", whitepixels)
        if (whitepixels<200000):
            print("This is not a ", colorarray[i])
            i = i+1
        else:
            print("This card is a ", colorarray[i])
            break
        cv.imshow('colors',res_effect)
        cv.waitKey(0)


    colors = cv.cvtColor(res_effect, cv.COLOR_HSV2BGR)
    cv.imshow('colors',colors)
    cv.waitKey(0)
    
    return(colorarray[i])
