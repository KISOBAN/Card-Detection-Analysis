import cProfile
import cv2 as cv
import numpy as np
import Cards
from PIL import Image
import mysql.connector
from Cards import getType
from Cards import getHash
from Cards import addCardtoDataBase
from matplotlib import pyplot as plt
import easyocr
import os
import glob
import imagehash


def getCards(image,contours,hierarchy,idx): 
    
        print(idx)
        cv.drawContours(image, contours, idx, (0,0,0), 3, cv.LINE_8, hierarchy, 0)
        rect = cv.minAreaRect(contours[idx])
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(image,[box],0,(0,0,255),2)

        
        rot_mat = cv.getRotationMatrix2D(rect[0], rect[2]-90, 1.0) # rotates the image so that it is upright
        result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
        x = int(rect[0][0])
        y = int(rect[0][1])
        
        w = int(rect[1][0]/2)
        h = int(rect[1][1]/2)
        cropped = result[y-w:y+w, x-h:x+h] 
        typeOfCard = getType(cropped)        
        print(typeOfCard)
        return (x,y,typeOfCard,cropped)

#Open the default camera
def preprocess(image): #applying a sharpening filter on the image 
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) #sharpening kernel
    im = cv.filter2D(image, -1, kernel)
    return im

cam = cv.VideoCapture(cv.CAP_DSHOW)

cam.set(cv.CAP_PROP_SETTINGS,1) 


# Get the default frame width and height
frame_width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cam.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret,thresh = cv.threshold(gray,127,255,cv.THRESH_BINARY)

    mask = np.ones(frame.shape[:2], dtype="uint8") * 255 #will be used later for separating the image

 
    edged = cv.Canny(thresh, 30, 200) 
    contours, hierarchy = cv.findContours(thresh,  cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE ) 
    drawing = np.zeros((edged.shape[0], edged.shape[1], 3), dtype=np.uint8)
    contouridx = np.array([])
    contourslength = np.array([])
    
    numofcards = 0
    img = 0
    imagearray = []
    typesarray = np.array([])
    for i in range(len(contours)):
        a = cv.contourArea(contours[i])
        if ((a>10000) & (hierarchy[0][i][3]!= -1)): # if the parent exists for that contour then it draws that contour 
           
            print(len(contours[i]))
            print(contours[i].shape)
            image = cv.drawContours(drawing, contours, i, (255,255,255), cv.FILLED, cv.LINE_8, hierarchy, 0)
            contouridx = np.append(contouridx,i) #appends the index of each contour into the index array so we can access each contour
            img = cv.bitwise_and(frame, drawing, mask) # this image is the resulting filtered out image

            x,y,typeofcard,cropped = getCards(img,contours,hierarchy,i) #writing the detected cards into a folder
            image = cv.putText(frame, typeofcard, (x-20,y+110), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 1, cv.LINE_AA)
            
            numofcards = numofcards + 1
            imagearray.append(cropped)
            typesarray = np.append(typesarray,typeofcard) 

    
    #img = cv.bitwise_and(frame, drawing, mask) # this image is the resulting filtered out image
   
    
    cv.imshow('camera feed', frame) #showing the original camera feed   
    cv.imshow('frame',edged)  #showing all the edges that are detected from the original feed from the canny edge detector

    cv.imshow('contours',drawing) #showing all the detected cards
    cv.imshow("filtered",img) #showing the filtered camera feed

    

    # Press 'q' to exit the loop
    if cv.waitKey(1) == ord('q'):
        break
        
# Release the capture and writer objects
cam.release()
cv.destroyAllWindows()


#finding out each card's type
hasharray = str(0)
print("Number of Cards Detected is:", numofcards)
for i in range(numofcards):
    
    cv.imwrite("DetectedCards/Card" + str(i) + ".png",imagearray[i]) #writes it to a folder
    
    name = getHash("DetectedCards/Card" + str(i) + ".png")
    hasharray = np.append(hasharray,str(name))
    
    addCardtoDataBase(str(name),typesarray[i])

    print("Image Hash:", name)
   
    cv.waitKey(0)
    



