import os
import cv2
from retinaface import RetinaFace

dirc = "my_face/unmasked/"

for filename in os.listdir(dirc):
    image_file = "my_face/unmasked/" +filename
    resp = RetinaFace.detect_faces(image_file)
    img = cv2.imread(image_file) 
    colour=(0, 255, 0)

    for f, v in resp.items():
        score_text = "s: " + str(v["score"])[:8]
        fa = v["facial_area"]
        cv2.putText(img, score_text, (fa[0], fa[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
        cv2.rectangle(img, (fa[2], fa[3]), (fa[0], fa[1]), colour, 1)

        for _, l in v["landmarks"].items():
            x = l[0].astype(int)
            y = l[1].astype(int)
            cv2.circle(img=img, center=(x, y), radius=3, color=colour, thickness=-1)

    print(image_file)
    cv2.imwrite("rf_output_" + filename, img)