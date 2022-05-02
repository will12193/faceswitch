import cv2
import os
import copy
import matplotlib.pyplot as plt
from retinaface import RetinaFace
from imutils import paths
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


# Load model
model=load_model(r'../training/custom_4370_32_100_v2.h5')

# Load image
image_file = r"./crowd1.jpg"
resp = RetinaFace.detect_faces(image_file)
image = cv2.imread(image_file) 
imageOutput = copy.deepcopy(image)

for i, f in resp.items():
    conf = f["score"]
    print(i + " - Confidence: ",conf)

    # Cut out detected face
    fa = f["facial_area"]
    face = image[fa[1]: fa[3], fa[0]: fa[2]]
    # plt.imshow(face)
    # plt.show()

    if conf < 0.9:
        label = 'Not a face?' 
        colour = (255, 0, 255)
    else:
        # Preprocess image for model
        face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
        face = cv2.resize(face,(96,96))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face,axis=0)
        
        # Use model to determine if mask is worn or not
        (withoutMask, mask) = model.predict(face)[0]
        print("MASK: ", mask)
        print("Without: ", withoutMask)

        # Colour and label for the output
        if mask > withoutMask:
            label = 'Mask' 
            colour = (0, 255, 0)
        else:
            label = 'No Mask'
            colour = (255, 0, 0)

        # Add probibility to label
        label="{}: {:.2f}%".format(label, max(mask,withoutMask) * 100)
        
    # Add label and bounding box to image
    cv2.putText(imageOutput, label, (fa[2], fa[3]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 2)
    cv2.rectangle(imageOutput, (fa[2], fa[3]), (fa[0], fa[1]), colour, 2)   

    # Add landmarks to face
    for _, l in f["landmarks"].items():
        x = l[0].astype(int)
        y = l[1].astype(int)
        cv2.circle(img=imageOutput, center=(x, y), radius=3, color=colour, thickness=-1)

plt.imshow(imageOutput)
plt.show()