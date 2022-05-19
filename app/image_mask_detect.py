import cv2
from matplotlib import pyplot as plt
from retinaface import RetinaFace
from tensorflow.keras.models import load_model
from retinaface_detect import detect_faces
from mask_detect import detect_mask

MODEL_PATH = r'./models/resNet50_3.h5'
 
def main():
    model = load_model(MODEL_PATH)
    rfModel = RetinaFace.build_model()

    frame = cv2.imread("../testing/unmasked.jpg")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find faces
    faces = detect_faces(frame, model = rfModel)

    # Detect mask
    frameDetect = detect_mask(frame, model, faces)
    plt.imshow(frameDetect)
    plt.show()


if __name__ == "__main__":
    main()