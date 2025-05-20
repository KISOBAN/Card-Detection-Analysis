import cv2 as cv
import numpy as np
import pytesseract
from PIL import Image
import mysql.connector
from matplotlib import pyplot as plt
from scipy.signal import wiener
import torch
import torchvision
import easyocr
import hashlib
import imagehash


def getType(img):
    colorarray = ["Link", "Effect", "Fusion", "Spell", "Synchro", "Xyz"]
    mask = cv.imread('Pictures/SampleCards/CardMask.jpg')
    mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    ret3,maskthresh = cv.threshold(mask,127,255,cv.THRESH_BINARY)
    
    size = img.shape[0:2] #accessing the dimensions of the image
    maskthresh = cv.resize(maskthresh, (size[1],size[0]), interpolation=cv.INTER_CUBIC)
   # cv.imshow("mask",maskthresh)
    #Cover the card image and saving it into a new image variable
    img1a = cv.bitwise_or(img,img,mask=maskthresh)

   # cv.imshow('maskedimage',img1a)
    #cv.waitKey(0)

    #Cropping out the Borders since the original size is 813x1185
    #img1a = img1a[26:1155, 26:783]

    HSVimg = cv.cvtColor(img1a, cv.COLOR_BGR2HSV)
   # cv.imshow('HSVimg',HSVimg)
    #cv.waitKey(0)

#B64B05 average effect card color [182,75,5] RGB
#Defining the range

#**************COLOR BOUNDARIES IN HSV FOR THE DIFFERENT COLORED CARDS THAT ARE AVAILABLE*************************************

    #Works for Blue Detects Ritual Monsters
    effect_lower = np.array([90,50,70], np.uint8) 
    effect_upper = np.array([120,255,255], np.uint8) #HSV notation 

    #works for Brown Detects Effect Monsters  RGB: Darkest Brown: [180,88,41] Lightest Brown: [230,202,190] -> [9  44 230]
    #[[[ 10 197 180]]]
    effect_lower1 = np.array([0,50,50], np.uint8) 
    effect_upper1 = np.array([10,255,255], np.uint8) #HSV notation 

    #works for fusion Monsters: RGB: Dark Purple:[130,55,46] -> [3 165 130], Light Purple: [222,199,228] -> [144  32 228]
    effect_lower2 = np.array([129,50,70], np.uint8) 
    effect_upper2 = np.array([158, 255, 255], np.uint8) #HSV notation 

    #works for spells: RGB Green:[0,255,0] ->  
    effect_lower3 = np.array([[80, 25, 25]], np.uint8) 
    effect_upper3 = np.array([100,255,255], np.uint8) #HSV notation 

    #works for synchro monsters: RGB 'Dark White':[222, 221, 219] -> [ 20   3 222] , 'Light White':[245, 244, 242] ->  [20   3 245]
    effect_lower4 = np.array([0, 0, 40], np.uint8) 
    effect_upper4 = np.array([180, 18, 230], np.uint8) #HSV notation 

    #works for XYZ monsters: RGB 'Black':[0,0,0] -> [0,0,0]  'Light Black:' [39, 38, 36]    ->  [20 20 39]
    #rgb(34, 34, 36) -> [120  14  36]       
    effect_lower5 = np.array([0,0,1], np.uint8) 
    effect_upper5 = np.array([180,255,30], np.uint8) #HSV notation 
#********************************************************************************************************************

    effectsmatrixlower = [effect_lower, effect_lower1, effect_lower2, effect_lower3, effect_lower4, effect_lower5]
    effectsmatrixupper = [effect_upper, effect_upper1, effect_upper2, effect_upper3, effect_upper4, effect_upper5]

    kernel = np.ones((5, 5), "uint8") 
    #Creating the Mask based off the boundaries

    whitepixels = np.array([])
    for i in range(len(colorarray)):
        low = effectsmatrixlower[i]
        high = effectsmatrixupper[i]

        effect_mask = cv.inRange(HSVimg, low, high) 
        effect_mask = cv.dilate(effect_mask, kernel) 
        #cv.imwrite('Mask'+ colorarray[i] + '.png',effect_mask)

        res_effect  = cv.bitwise_and(HSVimg, HSVimg, mask = effect_mask) 
      #  cv.imwrite('AND'+ colorarray[i] + '.png',res_effect)
        #cv.imshow(colorarray[i], effect_mask)
       # cv.waitKey(0)
        
        whitepixels = np.append(whitepixels,cv.countNonZero(effect_mask)) 
        #print("WhitePixels: ", whitepixels)
            
        #cv.imshow(colorarray[i], res_effect)
        #cv.waitKey(0)
    idx = whitepixels.argmax() #finding the index which corresponds to the most whitepixels
    #print("This card is a ", colorarray[idx], "type card")
    


    #Convert HSV back to BGR to see the color detection
    colors = cv.cvtColor(res_effect, cv.COLOR_HSV2BGR)
    #cv.imwrite('Convert.png',colors)
    #cv.imshow(colorarray[idx], colors)
    #cv.waitKey(0)
    return(colorarray[idx])



#*****************************************************************************************
                                 


def getHash(imgpath):
    image1 = Image.open(imgpath)
    hash1 = imagehash.average_hash(image1)
    return hash1

def addCardtoDataBase(name,color):
    mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "123456",
    database = "mydb"

   )
    mycursor = mydb.cursor()
    sql = "INSERT INTO cards VALUES (%s, %s); "
    val = (name,color)
    mycursor.execute(sql,val)
  
    mydb.commit()
 
  
