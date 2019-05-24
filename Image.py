import cv2
import numpy as np
from statistics import pvariance as pv

class SubFrame:
    def __init__(self,frame):
        self.frame = frame
        self.hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        self.lower_blue = np.array([111, 0, 0])
        self.upper_blue = np.array([169, 96, 255])
        self.w = 0
        self.h = 0
        self.x = 0
        self.y = 0
        self.mask = cv2.inRange(self.hsv, self.lower_blue, self.upper_blue)
        self.kernel = np.ones((5,5),np.uint8)
        self.dilation = cv2.dilate(self.mask,self.kernel,iterations = 2)
        self.kernel = np.ones((15,15),np.uint8)
        self.opening = cv2.morphologyEx(self.dilation, cv2.MORPH_OPEN, self.kernel)
        self.mask = cv2.morphologyEx(self.opening, cv2.MORPH_CLOSE, self.kernel) 
        self.res = cv2.bitwise_and(self.frame,self.frame, mask= self.mask)
        
    def contour(self):
        _ ,self.contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if len(self.contours) > 0:
            cnt = max(self.contours, key = cv2.contourArea)
            try:
                M = cv2.moments(cnt) 
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                self.x,self.y,self.w,self.h = cv2.boundingRect(cnt)
                self.frame = cv2.rectangle(self.frame,(self.x,self.y),(self.x+self.w,self.y+self.h),(0,255,0),2)
                #print("x : {0} Y : {1}".format(w,h))
            except:
                print('eror')
            if self.x+self.w/2 != 0 :
                cv2.circle(self.frame,(self.x+self.w/2,self.y+self.h/2), 12, (0,0,255), -1)
        
        return self.frame
    
    def CentreOfCircle(self):
        return self.x+self.w/2 ,self.y+self.h/2
    
    def ContourArea(self):
        return self.w*self.h
