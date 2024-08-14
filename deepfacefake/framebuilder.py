#!/usr/bin/env python
#!/usr/bin/python3
import os
import sys
from pathlib import Path
from sys import platform
from traceback import print_tb
from typing import Dict

import cv2
import onnx
import os
import json
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import landmark
from insightface.utils import face_align
from InsightFaceDectectRecognition import InsightFaceDectectRecognition

def getCurrentUserName():
    uname= os.environ.get('USERNAME')
    if uname==None or uname=="":
        uname= os.environ.get('USER')
        
    return uname

# insert at 1, 0 is the script path (or '' in REPL)
workingDir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, workingDir)

print("api mldlai working dir: " + workingDir)

if os.name == "nt":
    import msvcrt
else:
    import tty
    import termios
__isLinux = False
if platform == "linux" or platform == "linux2":
    __isLinux = True
    # linux
elif platform == "darwin":
    __isLinux = False
    # OS X
elif platform == "win32":
    __isLinux = False
    # Windows...


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#from insightface.data import get_image as ins_get_image

from skimage import transform as trans
#https://github.com/abhinavs95/model-zoo/blob/master/models/face_recognition/ArcFace/arcface_inference.ipynb

import onnxruntime

import blending

class OpenCvFrameBuilder:
    def __init__(self,workingDir) :
        self.workingDir=workingDir
        self.faceDetector= InsightFaceDectectRecognition(workingDir)
        pass
    
    def getFaceAreaFake(self,frame):
        padding= 0.25
        facesfound=self.faceDetector.DetectFace(frame)
        if len(facesfound)<=0:
            return  (None,None,None,None,None,None,None,None)
        (face,bbox,landmarkPts)=facesfound[0]
        (x,y,w,h)=bbox
        facecroped= self.faceDetector.CropPadding(frame,bbox,padding)
        areaface=[(x,y),(x+w,y)]
        areaface.extend(landmarkPts[:32])
        keepedMark= self.faceDetector.keepInsideArea(frame,areaface)
        xw=int( w*padding)
        yh=int(h*padding)
        x=x-xw
        y=y-yh
        w=w+xw+xw
        h=h+yh+yh
        bbox=(x,y,w,h)
        
        cropedorg= frame[y:y+h, x:x+w]
        keeped = keepedMark[y:y+h, x:x+w]
        
        return (keeped, face,bbox, landmarkPts,keepedMark,cropedorg, xw, yh)
        
    def process(self,frame, framefake):
        # oh,ow,oc=frame.shape
        # framefake= cv2.resize(frame,(ow,oh))
        # (oface,obbox,olandmarkPts)=self.faceDetector.DetectFace(frame)[0]
        # (ox,oy,ow,oh)=obbox      
        # oareaface=[(x,y),(x+w,y)]
        # oareaface.extend(olandmarkPts[:32])
        # okeepedMark= self.keepInsideArea(frame,oareaface)
        # facecroped= self.faceDetector.CropPadding(frame,bbox,0.01)
        # # self.drawLandmark(frame,landmarkPts)
        
        # # cv2.rectangle(frame, (x,y),(x+w,y+h), (0, 0, 255, 255),1)
        
        # areaface=[(x,y),(x+w,y)]
        # areaface.extend(landmarkPts[:32])
        (keeped, face,bbox,landmarkPts, keepdArea, croped, padx,pady)= self.getFaceAreaFake(frame)
                
        cv2.imwrite("keeped.png",keeped)
        x,y,w,h=bbox    
        
        
        (areafake,fakeface,fakebbox,fakelandmark,fakekeepedarea, fakecroped,fakepadx,fakepady)= self.getFaceAreaFake(framefake)
                
        areafake= cv2.resize(areafake, (w,h))        
        cv2.imwrite("areafake.png",areafake)
        # blended= self.blendImage(keeped, areafake)
        
        # (a,b,landmarkPts)=self.faceDetector.DetectFace(croped)[0]
        # (a,b,fakelandmark)=self.faceDetector.DetectFace(fakecroped)[0]
        
                # Compute the perspective transformation matrix
        # M = cv2.getPerspectiveTransform(np.array(landmarkPts, dtype='float32'),np.array( fakelandmark, dtype='float32'))

        # # Apply the perspective transformation to the target image
        # blended = cv2.warpPerspective(fakecroped, M, (croped.shape[1], croped.shape[0]))
        
        blended,landmark1,landmark2,bbox1,bbox2= blending.blendingImage(cv2.cvtColor( keeped, cv2.COLOR_BGRA2BGR),cv2.cvtColor( areafake, cv2.COLOR_BGRA2BGR))
    
        # oareaface=[(x,y),(x+w,y)]
        # oareaface.extend(landmark1[:32])
        # okeepedMark= self.faceDetector.keepInsideArea(blended,oareaface)
            
        self.drawOverlayImage(frame,blended,x,y)
        
        cv2.rectangle(frame, (x+padx,y+pady), (x+w-padx,y+h-pady), (0,0,255), 1)
        
    
        return frame
        # cv2.imwrite("finallblended.png",frame)
        
        # cv2.imshow("org", frame)
        # cv2.waitKey(0)
        pass
    def blendImage(self,imgsrc,imgoverlay):
                

        alpha = 0.1  # Weight of the first image (source)
        beta = 1 - alpha  # Weight of the second image (target)

        # Blend the images
        blended_image = cv2.addWeighted(imgsrc, alpha, imgoverlay, beta, 0)
        return blended_image
    
    def drawOverlayImage(self,frame,image1,x,y):
        
        # image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2BGRA)
        h1, w1,c = image1.shape
                
        # Define the region of interest (ROI) on image2
        roi = frame[y:y+h1, x:x+w1]
        
        roih,roiw, roic= roi.shape
        
                
        # Ensure that image1 is within the boundaries of image2
        if image1.shape[0] <= roih and image1.shape[1] <= roiw:
            # Overlay image1 onto the ROI of image2
            alpha_image1 = image1[..., 3] / 255.0 if image1.shape[2] == 4 else 1  # Use alpha channel if it exists
            alpha_image2 = 1.0 - alpha_image1
            
            for c in range(0, 3):
                roi[:, :, c] = (alpha_image1 * image1[:, :, c] + alpha_image2 * roi[:, :, c])
        else:
            print("The overlay image is larger than the region of interest.")
        
    def drawText(self,frame,text,xy,color = (255, 255, 255) ):                  
        # Define the font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Define font scale (size)
        font_scale = 0.5
        # Define color (BGR)
        # Define thickness of the text
        thickness = 1
        # Draw the text on the image
        cv2.putText(frame, text, xy, font, font_scale, color, thickness)
        pass
    def drawLandmark(self,frame,landmarkPts, color=(0, 0, 255, 255)):
        for idx, (x,y) in enumerate( landmarkPts):
            ## Convert to Scalar (in BGRA format)
            cv2.rectangle(frame, (x,y),(x+1,y+1), color,1)
            if idx<33:
                self.drawText(frame,f"{idx}",(x,y), color)
                
                
                
framebuilder=OpenCvFrameBuilder(workingDir)


frameorg= cv2.imread("/work/llm/Ner_Llm_Gpt/deepfacefake/ducmnd.jpg")
framefake= cv2.imread("/work/llm/Ner_Llm_Gpt/deepfacefake/hoandung.jpg")

# framebuilder.process(frameorg,framefake)

# 0 is the default camera (usually the built-in webcam)
cap = cv2.VideoCapture(0)
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Check if the frame was read correctly
    if not ret:
        print("Failed to grab frame")
        break
    try:
        framebuilder.process(frame,framefake)
    except Exception as ex:
        pass
    # Display the frame
    cv2.imshow('Camera Feed', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
