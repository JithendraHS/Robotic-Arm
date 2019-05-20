# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 12:25:38 2019

@author: Jithendra HS
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 23:14:06 2019

@author: Jithendra HS
"""
# import the necessary packages
from collections import deque
import numpy as np
import time
import speech_recognition as sr
import win32com.client
speaker = win32com.client.Dispatch("SAPI.SpVoice")
import argparse
import imutils
import cv2
import urllib #for reading image from URL
import serial
import struct
import math
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours

ser= serial.Serial('COM5',9600)

r=sr.Recognizer()
with sr.Microphone() as source:
    speaker.Speak("hi Sir")
    print('Hi Sir')
    speaker.Speak("I can pick object for you")
    print('I can pick object for you')
    speaker.Speak("Choose below named colored object to pick")
    print('Choose below named colored object to pick')
    speaker.Speak("red object")
    print('RED object')
    speaker.Speak("green object")
    print('GREEN object')
    speaker.Speak("blue object")
    print('BLUE object')
    speaker.Speak("yellow object")
    print('YELLOW object')
    speaker.Speak("orange object")
    print('ORANGE object')
    speaker.Speak("i am listening")
    print('i am listening...')
    audio=r.listen(source)
    speaker.Speak("done")
    print('Done')
try:
   #text=r.recognize_google(audio)
   text="red";
   speaker.Speak("you said "+ text)
   print('you said: '+text)
except Exception as e:
    print(e)
    
RADIANS = 0
DEGREES = 1

l1 = 17 #Length of link 1
l2 = 15#length of link 2
l3 = 9  #length of link 3

#This is the constant angle l3 must maintain
#with the ground plane. Must be in degrees
l3ang = 0
# define the lower and upper boundaries of the colors in the HSV color space
#lower = {'red':(166, 84, 141), 'green':(66, 122, 129), 'blue':(97, 100, 117), 'yellow':(23, 59, 119), 'orange':(0, 50, 80)} #assign new item lower['blue'] = (93, 10, 0)
#upper = {'red':(186,255,255), 'green':(86,255,255), 'blue':(117,255,255), 'yellow':(54,255,255), 'orange':(20,255,255)}
if text == "red":
    lower = {'red':(166, 84, 141)}
    upper = {'red':(186,255,255)}
if text == "green":
    lower = {'green':(66, 122, 129)}
    upper = {'green':(86,255,255)} 
if text == "blue":
    lower = {'blue':(97, 100, 117)}
    upper = {'blue':(117,255,255)}
if text == "yellow":
    lower = {'yellow':(23, 59, 119)}
    upper = {'yellow':(54,255,255)} 
if text == "orange":
    lower = {'orange':(0, 50, 80)}
    upper = {'orange':(20,255,255)}     
# define standard colors for circle around the object
colors = {'red':(0,0,255), 'green':(0,255,0), 'blue':(255,0,0), 'yellow':(0, 255, 217), 'orange':(0,140,255)}
img_counter = 0

count=0
count1=0
x1=0
y1=0
img_name1=0

camera = cv2.VideoCapture(1)
#cv2.namedWindow("frame")

# keep looping
#while True:
while count != 110:
    count+=1
    count1+=1
    # grab the current frame
    (grabbed, frame) = camera.read()
    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    # if args.get("video") and not grabbed:
    #      break
    # resize the frame, blur it, and convert it to the HSV
    # color space
    frame = imutils.resize(frame, width=900)

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    #for each color in dictionary check object in frame
    for key, value in upper.items():
        # construct a mask for the color from dictionary`1, then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        kernel = np.ones((9,9),np.uint8)
        mask = cv2.inRange(hsv, lower[key], upper[key])
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                
        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None
        
        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            
            c = max(cnts, key=cv2.contourArea)
            #((x, y), radius) = cv2.minEnclosingCircle(c)
            (x, y, w, h) = cv2.boundingRect(c)
            #if x//20==w//20 & h//20==y1//20:
            print(x,y)
            #print(y)
            #print(w)
            #print(h)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
            # only proceed if the radius meets a minimum size. Correct this value for your obect's size
            #if radius > 0.5:
            if x & y > 0.5:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
               # cv2.circle(frame, (int(x), int(y)), int(radius), colors[key], 2)
                #cv2.putText(frame,key + " ball", (int(x-radius),int(y-radius)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,colors[key],2)
                cv2.rectangle(frame, (int(x), int(y)),(int(x+w),int(y+h)), colors[key], 2)
                cv2.putText(frame,key + "object", (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,colors[key],2)
                x1=x
                y1=y
    val=int(x1)
    val1=val//255
    val2=val%255
    print(val1)
    print(val2)
     
    # show the frame to our screen
    cv2.imshow("Frame", frame)
    
    if count==100:
        #count=0;
        cv2.imwrite('img_name1.jpg', frame)
        time.sleep(2)
        
# cleanup the camera and close any open windows
camera.release()
#cv2.destroyAllWindows()"""

num=0
while num!=10000:
    num+=1
    
img_counter=0
#def main():

   # cap = cv2.VideoCapture(0)
#image = cv2.imread('img2.jpg')
image = cv2.imread('img_name1.jpg')    
    #if cap.isOpened():
     #   ret, frame = cap.read()
    #else:
     #   ret = False
        
   # while ret:
     #   ret, frame = cap.read()
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
if text == "blue":
            #Blue color
            low = np.array([100, 50, 50])
            high = np.array([140, 255, 255])
            
            
if text == "green":   
            #Greeen color
            low = np.array([66, 122, 129])
            high = np.array([86,255,255])
            
if text == "red": 
            #Red color
            low = np.array([166, 84, 141])
            high = np.array([186, 255, 255])
if text == "yellow":
            #Yellow color
            low = np.array([23, 59, 119])
            high = np.array([54,255,255]) 
if text == "orange":
            #Orange color
            low = np.array([0, 50, 80])
            high = np.array([20,255,255])
            
else:
            low = np.array([255, 255, 255])
            high = np.array([255, 255, 255])
        
image_mask = cv2.inRange(hsv, low, high)
output = cv2.bitwise_and(image,image, mask = image_mask)
cv2.imshow("Image mask",image_mask)
cv2.imshow("original webcam feed", image)
cv2.imshow("color tracking", output)

cv2.imwrite('img_name2.jpg', output)
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

width = 16
time.sleep(2)
# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread('img_name2.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable
#(cnts,_) = contours.sort_contours(cnts)
pixelsPerMetric = None

ang0=0
ang1=0
ang2=0
# loop over the contours individually
for c in cnts:
	# if the contour is not sufficiently large, ignore it
    if cv2.contourArea(c) < 10:
        continue

	# compute the rotated bounding box of the contour
     
    orig = image.copy()
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

	# order the points in the contour such that they appear
	# in top-left, top-right, bottom-right, and bottom-left
	# order, then draw the outline of the rotated bounding
	# box
    box = perspective.order_points(box)
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

	# loop over the original points and draw them
    for (x, y) in box:
        cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
          
	# unpack the ordered bounding box, then compute the midpoint
	# between the top-left and top-right coordinates, followed by
	# the midpoint between bottom-left and bottom-right coordinates
    (tl, tr, br, bl) = box
    #print(tl)
    #print(tr)
    #print(bl)
    #print(br)
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

	# compute the midpoint between the top-left and top-right points,
	# followed by the midpoint between the top-righ and bottom-right
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

	# draw the midpoints on the image
    cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

	# draw lines between the midpoints
    cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
		(255, 0, 255), 2)
    cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
		(255, 0, 255), 2)

	# compute the Euclidean distance between the midpoints
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

	# if the pixels per metric has not been initialized, then
	# compute it as the ratio of pixels to supplied metric
	# (in this case, inches)
    if pixelsPerMetric is None:
		#pixelsPerMetric = dB / args["width"]
         pixelsPerMetric = dB / width
        #print(pixelsPerMetric)
	# compute the size of the object
    dimA = dA / pixelsPerMetric
    dimB = dB / pixelsPerMetric
    print((tltrX+blbrX)/(2*pixelsPerMetric))
    print((tlblY+trbrY)/(2*pixelsPerMetric))
    x1=int((tltrX+blbrX)/(2*pixelsPerMetric))
    y1=int((tlblY+trbrY)/(2*pixelsPerMetric))
    print("coordinates are")
    print((tltrX+blbrX)/(2*pixelsPerMetric))
    print((tlblY+trbrY)/(2*pixelsPerMetric))
    
	# draw the object sizes on the image
    cv2.putText(orig, "{:.1f}cm".format(dimA),
		(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)
    cv2.putText(orig, "{:.1f}cm".format(dimB),
		(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)

	# show the output image
    cv2.imshow("Image", orig)
    cv2.imwrite('img_name3.jpg',orig)
    time.sleep(1)
x=x1
print(x)
z=y1
print(z)
y=9
#def invkin3(x, y, z, angleMode=DEGREES):
"""Returns the angles of the first two links and
     the base drum in the robotic arm as a list.
    returns -> (th0, th1, th2)
    
    x - The x coordinate of the effector
    y - The y coordinate of the effector
    z - The z coordinate of the effector
    angleMode - tells the function to give the angle in
                degrees/radians. Default is degrees
    output:
    th0 - angle of the base motor
    th1 - angle of the first link w.r.t ground
    th2 - angle of the second link w.r.t the first"""
th0 = math.atan2(z,x)
x = (x**2 + z**2)**0.5
    #stuff for calculating th2
r_2 = x**2 + y**2
l_sq = l1**2 + l2**2
term2 = (r_2 - l_sq)/(2*l1*l2)
   # term2=1
term1 =abs(((1 - term2**2)**0.5)*-1)
    #calculate th2
th2 = math.atan2(term1, term2)
    #optional line. Comment this one out if you 
    #notice any problems
#th2 = -1*th2

    #Stuff for calculating th2
k1 = l1 + l2*math.cos(th2)
k2 = l2*math.sin(th2)
   # r  = (k1**2 + k2**2)**0.5
gamma = math.atan2(k2,k1)
    #calculate th1
th1 = math.atan2(y,x) - gamma
print("degrees are") 
ang0=int(math.degrees(th0))
ang1=int(math.degrees(th1)) 
ang2=int(math.degrees(th2))
print(90-ang0)
print(abs(ang1))
print(180-ang2)
#ser= serial.Serial('COM5',9600)    
"""
    if(angleMode == RADIANS):
        return int(th0), int(th1), int(th2)
    else:
        return int(math.degrees(th0)), int(math.degrees(th1)),\
            int(math.degrees(th2))
        return math.degrees(th0), math.degrees(th1),\
            math.degrees(th2)  """  
#if __name__ == "__main__":
    #print(invkin3(x1, y1, 8, DEGREES))
"""
x=x1
y=2
z=y1
a=4
b=17
c=13
theta1=int(math.degrees(math.atan(y/x)))
print("theta1")
print(theta1)
r1=abs(math.sqrt(y*y+x*x))
print(r1)
r2=abs(z-a)
print(r2)
si2= int(math.degrees(math.atan(r2/r1)))
print(si2)
r3=abs(math.sqrt(r1*r1+r2*r2))
print(r3)
si1=int(math.degrees(math.acos((c*c-b*b-r3*r3)/abs(-2*b*r3))))
print(si1)
theta2=si2-si1
print("theta2")
print(theta2)
si3=int(math.degrees(math.acos((r3*r3-b*b-c*c)/abs(-2*b*c))))
print(si3)
theta3=180-si3
print("theta3")
print(theta3)
ser.write(struct.pack('>H',int(abs(theta1))))
#time.sleep(1000)
ser.write(struct.pack('>H',int(abs(theta2))))
#time.sleep(1000)
ser.write(struct.pack('>H',int(abs(theta3))))
#time.sleep(1000)
"""
# if count1==300:
ser.write(struct.pack('>H',int(abs(90-ang0))))
ser.write(struct.pack('>H',int(abs(ang1))))
ser.write(struct.pack('>H',int(abs(180-ang2))))
   ###     count1=0
    

key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
if key == ord("q"):
    
#cv2.waitKey(0)
    cv2.destroyAllWindows()
