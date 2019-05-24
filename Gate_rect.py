import cv2
import numpy as np
from time import time

from Image import SubFrame

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS,30)
_, frame = cap.read()
img = np.zeros((480,640,3), np.uint8)
wide = frame.shape[1]
height = frame.shape[0]
Nslice = 10
subframew = wide / Nslice
subframeh = height / Nslice
nclus = 4

while(1):
    a = time()
    _, frame = cap.read()
    framel=[]
    for i in range(Nslice):
        framel.append(list())
    _, frame = cap.read()
    obj , X , Subframe=[] , [] , []
    w , w1 , h , h1 = [] , [] , [] , []
    for i in range(Nslice**2):
        w.append(0)
        h.append(0)

    for i in range(Nslice):
        for j in range(Nslice):
            framel[i].append(frame[i*subframeh:(i+1)*subframeh , j*subframew:(j+1)*subframew].copy())  
            obj.append(SubFrame(framel[i][j]))
            Subframe.append(obj[(i*Nslice)+j].contour())
            img[i*subframeh:(i+1)*subframeh ,j*subframew:(j+1)*subframew] = Subframe[(i*Nslice)+j].copy()
            w[(i*Nslice)+j] , h[(i*Nslice)+j] = obj[(i*Nslice)+j].CentreOfCircle()
            h1.append(h[(i*Nslice)+j]+(subframeh*i))
            w1.append(w[(i*Nslice)+j]+(subframew*j))    
    xdata , ydata = [] , []
    for i in range(Nslice**2):
        if w[i] and h[i] is not 0:
            X.append([w1[i],h1[i]])
            xdata.append(w1[i])
            ydata.append(h1[i])
    


    try:
        cv2.circle(frame,(int(min(xdata)),int(min(ydata))), 12, (200,0,0), -1)
        cv2.circle(frame,(int(min(xdata)),int(max(ydata))), 12, (200,0,0), -1)
        cv2.circle(frame,(int(max(xdata)),int(min(ydata))), 12, (200,0,0), -1)
        cv2.circle(frame,(int(max(xdata)),int(max(ydata))), 12, (200,0,0), -1)
        cv2.circle(frame,( int( (max(xdata)+min(xdata))/2 ),int( (max(ydata)+min(ydata))/2)), 12, (120,0,120), -1)
    except Exception as e :
        print("Error : , " , e)
    for i in range(1,Nslice):
        cv2.line(img,(subframew*i,0),(subframew*i,height),(255,255,255),1)
        cv2.line(img,(0,subframeh*i),(wide,subframeh*i),(255,255,255),1)
    cv2.imshow('frame',frame)
    cv2.imshow('img',img)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    #print((time()-a)*30)

cap.release()
cv2.destroyAllWindows()
