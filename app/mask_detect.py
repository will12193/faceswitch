import cv2
import copy
import time
from retinaface import RetinaFace
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from retinaface_detect import detect_faces

MODEL_PATHS = [
    # r'./models/maskdetection_1.h5', 
    # r'./models/maskdetection_2.1.h5', 
    r'./models/maskdetection_3.h5', 
    r'./models/maskdetection_3_60.h5', 
    # r'./models/resNet50_1.h5', 
    # r'./models/resNet50_1_60.h5', 
    # r'./models/resNet50_2.1.h5', 
    r'./models/resNet50_3.h5',
    r'./models/resNet50_3_60.h5'
]

# Takes an CV2 image and outputs the same image with faces bounded and 
# an indication as to whether they are wearing masks
def detect_mask(image, model, faces, verbose=False):
    if isinstance(faces, tuple):
        return image

    imageOutput = copy.deepcopy(image)
    for i, f in faces.items():
        conf = f["score"]
        if verbose:
            print(i + " - Confidence: ",conf)

        # Cut out detected face
        fa = f["facial_area"]
        face = image[fa[1]: fa[3], fa[0]: fa[2]]

        if conf < 0.9:
            label = 'Not a face?' 
            colour = (255, 0, 255)
        else:
            # Preprocess image for model
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (96, 96))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face,axis=0)
            
            # Use model to determine if mask is worn or not
            if verbose:
                startTime = time.time()

            (mask, withoutMask) = model.predict(face)[0]

            if verbose:
                endTime = time.time()
                print("Model runtime: "+ str(endTime - startTime))
                print("MASK: ", mask)
                print("WITHOUT: ", withoutMask)

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

    return imageOutput

def main():
    models = []
    for m in MODEL_PATHS:
        models.append(load_model(m))
    rfModel = RetinaFace.build_model()

    cap = cv2.VideoCapture(0)
    if (cap.isOpened()== False):
        raise Exception("Error opening VideoCapture")

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            # Find faces
            faces = detect_faces(frame, model = rfModel)

            # Detect mask
            for i, m in enumerate(MODEL_PATHS):
                # print(m)
                # frameDetect = detect_mask(frame, models[i], faces, verbose=True)
                frameDetect = detect_mask(frame, models[i], faces)
                cv2.imshow(m, frameDetect)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else: 
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()