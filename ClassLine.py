import cv2
import numpy as np
from time import time
from sklearn.cluster import KMeans
from sklearn import metrics

class Line:
    def __init__(self,image,frame):
        self.hsv = image
        self.frame = frame
        self.lower_blue = np.array([101, 217, 0])
        self.upper_blue = np.array([ 126, 255, 255])
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
        self.contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
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

cap = cv2.VideoCapture(1)
w=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
h=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
w1=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
h1=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
img = np.zeros((480,640,3), np.uint8)


while(1):
    t1 = time()
    framel=[[],[],[],[]]
    hsvl=[[],[],[],[]]
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for i in range(0,4):
        for j in range(0,4):
            framel[i].append(frame[i*480/4:(i+1)*480/4 , j*640/4:(j+1)*640/4].copy())  
            hsvl[i].append(hsv[i*480/4:(i+1)*480/4 , j*640/4:(j+1)*640/4].copy())
    #print(len(framel),len(hsvl))

    LINEa0 = Line(hsvl[0][0],framel[0][0])
    framea0= LINEa0.contour()

    LINEa1 = Line(hsvl[0][1],framel[0][1])
    framea1 = LINEa1.contour()

    LINEa2 = Line(hsvl[0][2],framel[0][2])
    framea2 = LINEa2.contour()
    
    LINEa3 = Line(hsvl[0][3],framel[0][3])
    framea3 = LINEa3.contour()

    LINEb0 = Line(hsvl[1][0],framel[1][0])
    frameb0 = LINEb0.contour()

    LINEb1 = Line(hsvl[1][1],framel[1][1])
    frameb1 = LINEb1.contour()

    LINEb2 = Line(hsvl[1][2],framel[1][2])
    frameb2 = LINEb2.contour()

    LINEb3 = Line(hsvl[1][3],framel[1][3])
    frameb3 = LINEb3.contour()

    LINEc0 = Line(hsvl[2][0],framel[2][0])
    framec0 = LINEc0.contour()

    LINEc1 = Line(hsvl[2][1],framel[2][1])
    framec1 = LINEc1.contour()

    LINEc2 = Line(hsvl[2][2],framel[2][2])
    framec2 = LINEc2.contour()
    
    LINEc3 = Line(hsvl[2][3],framel[2][3])
    framec3 = LINEc3.contour()

    LINEd0 = Line(hsvl[3][0],framel[3][0])
    framed0 = LINEd0.contour()

    LINEd1 = Line(hsvl[3][1],framel[3][1])
    framed1 = LINEd1.contour()

    LINEd2 = Line(hsvl[3][2],framel[3][2])
    framed2 = LINEd2.contour()
    
    LINEd3 = Line(hsvl[3][3],framel[3][3])
    framed3 = LINEd3.contour()
    #hi
    img[ 0:480/4 ,         0:640/4 ] =       framea0.copy()
    img[ 0:480/4 ,   640/4:640/4*2 ] =   framea1.copy()
    img[ 0:480/4 , 640/4*2:640/4*3 ] = framea2.copy()
    img[ 0:480/4 , 640/4*3:640/4*4 ] =   framea3.copy()

    img[ 480/4:480/4*2 ,         0:640/4 ] =   frameb0.copy()
    img[ 480/4:480/4*2 ,   640/4:640/4*2 ] = frameb1.copy()
    img[ 480/4:480/4*2 , 640/4*2:640/4*3 ] =     frameb2.copy()
    img[ 480/4:480/4*2 , 640/4*3:640/4*4 ] = frameb3.copy()

    img[ 480/4*2:480/4*3 ,         0:640/4 ] = framec0.copy()
    img[ 480/4*2:480/4*3,    640/4:640/4*2 ] =   framec1.copy()
    img[ 480/4*2:480/4*3 , 640/4*2:640/4*3 ] =       framec2.copy()
    img[ 480/4*2:480/4*3 , 640/4*3:640/4*4 ] =   framec3.copy()

    img[ 480/4*3:480/4*4 ,         0:640/4 ] = framed0.copy()
    img[ 480/4*3:480/4*4 ,   640/4:640/4*2 ] =   framed1.copy()
    img[ 480/4*3:480/4*4 , 640/4*2:640/4*3 ] =       framed2.copy()
    img[ 480/4*3:480/4*4 , 640/4*3:640/4*4 ] =   framed3.copy()
    
    w[0] , h[0] = LINEa0.CentreOfCircle()  
    w[1] , h[1] = LINEa1.CentreOfCircle()  
    w[2] , h[2] = LINEa2.CentreOfCircle() 
    w[3] , h[3] = LINEa3.CentreOfCircle()
    
    w1[0] = w[0]
    w1[1] = w[1] + 640/4 
    w1[2] = w[2] + 640/4*2 
    w1[3] = w[3] + 640/4*3 

    h1[0] = h[0] 
    h1[1] = h[1] 
    h1[2] = h[2] 
    h1[3] = h[3] 

    w[4] , h[4] = LINEb0.CentreOfCircle()
    w[5] , h[5] = LINEb1.CentreOfCircle()
    w[6] , h[6] = LINEb2.CentreOfCircle()
    w[7] , h[7] = LINEb3.CentreOfCircle()

    h1[4] = h[4] + 480/4
    h1[5] = h[5] + 480/4
    h1[6] = h[6] + 480/4
    h1[7] = h[7] + 480/4

    w1[4] = w[4]
    w1[5] = w[5] + 640/4
    w1[6] = w[6] + 640/4*2 
    w1[7] = w[7] + 640/4*3

    w[8] , h[8] = LINEc0.CentreOfCircle()
    w[9] , h[9] = LINEc1.CentreOfCircle()
    w[10] , h[10] = LINEc2.CentreOfCircle()
    w[11] , h[11] = LINEc3.CentreOfCircle()
    
    h1[8] =h[8] + 480/4*2
    h1[9] = h[9] +480/4*2
    h1[10] = h[10] +480/4*2
    h1[11] = h[11] +480/4*2
    
    w1[8] = w[8]
    w1[9]  = w[9] +640/4
    w1[10] = w[10]+640/4*2 
    w1[11] = w[11] + 640/4*3

    w[12] , h[12] = LINEd0.CentreOfCircle()
    w[13] , h[13] = LINEd1.CentreOfCircle()
    w[14] , h[14] = LINEd2.CentreOfCircle()
    w[15] , h[15] = LINEd3.CentreOfCircle()
    
    h1[12] = h[12] +480/4*3
    h1[13] = h[13] +480/4*3
    h1[14] = h[14] +480/4*3
    h1[15] = h[15] +480/4*3

    w1[12] = w[12]
    w1[13] = w[13] +640/4
    w1[14] = w[14] +640/4*2 
    w1[15] = w[15] +640/4*3

    
    X = []
    for i in range(16):
        if w[i] and h[i] is not 0:
            X.append([w1[i],h1[i]])
    #print('X : {0}'.format(X))
    #cv2.circle(frame,(640/4,480/4), 12, (255,0,0), -1)
    try:
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        #print(kmeans.labels_)

        for i in range(len(kmeans.labels_)):        
            print("entrytry")
            if kmeans.labels_[i] == 0 :
                print("entry0")
                #cv2.circle(frame,(X[i][0],X[i][1]), 12, (0,0,255), -1)

            if kmeans.labels_[i] == 1 :
                print("entry1")
                #cv2.circle(frame,(X[i][0],X[i][1]), 12, (0,255,0), -1)
        #print(type(kmeans.cluster_centers_))
        cv2.circle(frame,(int(kmeans.cluster_centers_[0][0]),int(kmeans.cluster_centers_[0][1])), 12, (255,0,0), -1)
        cv2.circle(frame,(int(kmeans.cluster_centers_[1][0]),int(kmeans.cluster_centers_[1][1])), 12, (255,255,0), -1)
    except:
        print("no data")
    
    
    '''cv2.circle(frame,(320,80), 12, (0,0,255), -1)
    cv2.circle(frame,(320,240), 12, (0,0,255), -1)
    cv2.circle(frame,(320,400), 12, (0,0,255), -1)'''
    #print("Weight 0 : " ,320 - w[0])

    cv2.imshow('frame',frame)
    cnt = 0
    '''for i in range(0,3):
        for j in range(0,3):
            cv2.imshow('frame'+str(cnt),framel[i][j])
            cnt += 1'''
    
    cv2.line(img,(640/4,0),(640/4,480),(255,255,255),1)
    cv2.line(img,(640/4*2,0),(640/4*2,480),(255,255,255),1)
    cv2.line(img,(640/4*3,0),(640/4*3,480),(255,255,255),1)

    cv2.line(img,(0,480/4),(640,480/4),(255,255,255),1)
    cv2.line(img,(0,480/4*2),(640,480/4*2),(255,255,255),1)
    cv2.line(img,(0,480/4*3),(640,480/4*3),(255,255,255),1)

    cv2.imshow('frame12',img)

    #print("w: {0} H : {1}".format(w,h))
    k = cv2.waitKey(1) & 0xFF
    #print(time() - t1)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
