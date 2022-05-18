import cv2
from retinaface import RetinaFace
from tensorflow.keras.models import load_model
from retinaface_detect import detect_faces
from mask_detect import detect_mask

MODEL_PATH = r'./models/resNet50_3.h5'
 
def main():
    model = load_model(MODEL_PATH)
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
            frameDetect = detect_mask(frame, model, faces)
            cv2.imshow(MODEL_PATH, frameDetect)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else: 
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()