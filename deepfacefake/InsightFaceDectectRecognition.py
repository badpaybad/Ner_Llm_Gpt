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

class ImageBBox(object):
    def __init__(self) :
        self.bbox=[]
        self.kps=[]
        #self.shape = (5, 2)
        pass
    pass
    
class InsightFaceDectectRecognition:
    def __init__(self,workingDir) :
        self.workingDir=workingDir
        
        print("InsightFaceDectectRecognition workingdir: "+ workingDir)
                        
        self.detector = FaceAnalysis(allowed_modules=['detection', 'landmark_2d_106'], root=f"{workingDir}/weights/")
        self.detector.prepare(ctx_id=0, det_size=(320, 320))
        print("self.detector")
        print(self.detector)
        
    def DetectFace(self,frameRgb):
        """[(facecrop,(x,y,w,h),[pointlandmark])]
        
        facecrop use for VectorFace
        Args:
            frameRgb ([type]): [description]
        """
        
        faces =  self.detector.get(frameRgb)
        #print("faces", len(faces))
        listFound=[]
        for face in faces:
            #print(face)
            x,y,x1,y1= face.bbox
            x,y,x1,y1=int(x),int(y),int(x1),int(y1)
            #cv2.rectangle(tim,(int(x),int(y)),(int(x1),int(y1)),(0,0,255,0))
            
            lmk = face.landmark_2d_106
            lmk = np.round(lmk).astype(np.int32)
            landmarkPts=[]
            
            for i in range(lmk.shape[0]):
                p = tuple(lmk[i])
                landmarkPts.append(p)
                #cv2.circle(tim, p, 1, color, 1, cv2.LINE_AA)
            
            listFound.append((face,(x,y,x1-x,y1-y),landmarkPts))
            
            # 0->32: vien bao tu tai trai -> cam -> tai phai
            # 33 -> 42 : mat phai
            # 43-51: long may phai
            # 52 -> 71 : moi 
            # 72 -> 86 : mui
            # 87 -> 96: mat trai
            # 97 -> 105: long may trai
            #
            
        del faces
        return listFound
    
    def CropPadding(self,cv2img,faceregion, padding=0.25):
        x,y,w,h=faceregion
        
        deltax=int(w*padding)
        deltay=int(h*padding)
        x=x-deltax
        y=y-deltay
        w=w+deltax+deltax
        h=h+deltay+deltay
        
        return cv2img[y:y+h,x:x+w]
            
    def AlignFace(self,frameRgb, facecrop):
        aimg = face_align.norm_crop(frameRgb, landmark=facecrop.kps)
        return aimg
                
    def keepInsideArea(self,frame, points):
        sortClw= self.sortPointsClockwise(points)
        sortClw = np.array(sortClw)
                
        # Convert the frame to RGBA (add an alpha channel)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        # # Create a mask with the same dimensions as the frame, initialized to zeros (black)
        # mask = np.zeros_like(frame)
                
        # Create a mask with the same dimensions as the frame, initialized to zeros (black)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        # # Fill the polygon defined by 'points' with white color (255) on the mask
        # cv2.fillPoly(mask, [sortClw], (255, 255, 255))
        cv2.fillPoly(mask, [sortClw], 255)
        frame[mask == 0, 3] = 0
        
        # If you want to remove the black background entirely and replace it with transparency:
        background = np.zeros_like(frame)  # transparent background
        masked_frame = np.where(mask[..., None] == 0, background, frame)

        # # Apply the mask to the frame using bitwise AND to keep only the area inside the polygon
        # masked_frame = cv2.bitwise_and(frame, mask)

        # # Optionally, convert masked areas to black (if needed)
        # # background = np.ones_like(frame) * 0  # black background
        # background = np.ones_like(frame)  # transparent background
        # masked_frame = np.where(mask == 0, background, masked_frame)
        masked_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2BGRA)
        
        return masked_frame
    def sortPointsClockwise(self, points):
                
        # Calculate the centroid of the points
        centroid = np.mean(points, axis=0)

        # Sort points by the angle relative to the centroid
        def angle_from_centroid(point):
            return np.arctan2(point[1] - centroid[1], point[0] - centroid[0])

        sorted_points = sorted(points, key=angle_from_centroid)

        # Convert to numpy array after sorting (for OpenCV functions)
        sorted_points = np.array(sorted_points, dtype=np.int32)
        
        return sorted_points
    def sortPointsTopDown(self, points):
        # Sorting the list by y values (ascending)
        sorted_points = sorted(points, key=lambda point: point[1])
        return sorted_points
    
    def landmarkSegments(self, landmarkPts):
        result=[]
        result.append(landmarkPts[:33])
        result.append(landmarkPts[33:43])
        result.append(landmarkPts[43:52])
        result.append(landmarkPts[52:72])
        result.append(landmarkPts[72:87])
        result.append(landmarkPts[87:97])
        result.append(landmarkPts[97:])
        return result
    
    def getBbox(self, points):
        return cv2.boundingRect(np.array( points))

    
