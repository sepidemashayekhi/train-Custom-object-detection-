from matplotlib import image
import tensorflow as tf 
import os
import numpy as np 
import cv2
print(' ....Start Run Code ... ')

class Detect:
    def __init__(self) :
        classpath='G:/my_project/train-Custom-object-detection-/classlist.txt'
        Dirpath='G:/my_project/train-Custom-object-detection-/my_mobilenet_model/saved_model'
        
        with open(classpath,'r') as f:
            self.classList=f.read().split(',') 
        print(self.classList)
        self.ColorList=np.random.uniform(low=0,high=255,size=(len(self.classList)-1,3))
        tf.keras.backend.clear_session()
        self.model=tf.saved_model.load(Dirpath)
        print("Load models .....")
    def creatbondingBox(self,image,threshold=0.5):
        inputTensor=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        inputTensor=tf.convert_to_tensor(inputTensor)
        inputTensor=inputTensor[tf.newaxis,...]
        detection=self.model(inputTensor)
        bboxs=detection['detection_boxes'][0].numpy()
        classIndexes=detection['detection_classes'][0].numpy().astype(np.int32)
        classeScores=detection['detection_scores'][0].numpy()
        

        imH, imW, imC=image.shape
        bboxInd=tf.image.non_max_suppression(bboxs,classeScores,max_output_size=50,
                                             iou_threshold=threshold,score_threshold=threshold)
        

        if len(bboxInd) !=0:
            for i in bboxInd:
                bbox=tuple(bboxs[i].tolist())
                classConfidence=round(100*classeScores[i])
                classIndex=classIndexes[i]
                classLabelText=self.classList[classIndex-1]
                classColor=self.ColorList[classIndex-1]
                displayText='{} :{} %'.format(classLabelText,classConfidence)
                
                ymin,xmin,ymax,xmax=bbox
                ymin, xmin, ymax, xmax=(ymin*imH,xmin*imW,ymax*imH,xmax*imW)
                ymin, xmin, ymax, xmax=int(ymin),int(xmin),int(ymax),int(xmax)
                cv2.rectangle(image,(xmin,ymin),(xmax,ymax),color=classColor,thickness=1)
                cv2.putText(image,displayText,(xmin,ymin-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,classColor,2)

                lineWidth=min(int((xmax-xmin)*0.2),int((ymax-ymin)*0.2))

                cv2.line(image,(xmin,ymin),(xmin+lineWidth,ymin),color=classColor,thickness=5)
                cv2.line(image, (xmin, ymin), (xmin , ymin+lineWidth), color=classColor, thickness=5)

                cv2.line(image, (xmax, ymin), (xmax - lineWidth, ymin), color=classColor, thickness=5)
                cv2.line(image, (xmax, ymin), (xmax, ymin+ lineWidth), color=classColor, thickness=5)
                #################################
                cv2.line(image, (xmin, ymax), (xmin + lineWidth, ymax), color=classColor, thickness=5)
                cv2.line(image, (xmin, ymax), (xmin, ymax -lineWidth), color=classColor, thickness=5)

                cv2.line(image, (xmax, ymax), (xmax - lineWidth, ymax), color=classColor, thickness=5)
                cv2.line(image, (xmax, ymax), (xmax, ymax - lineWidth), color=classColor, thickness=5)
            
            return image


    
detect=Detect()
image=cv2.imread('F:/datasets/Nurse/dataset/test/saf.jpg')
image=detect.creatbondingBox(image)
cv2.imshow("image",image)
cv2.waitKey(0)