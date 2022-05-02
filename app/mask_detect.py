import cv2
import copy
from retinaface import RetinaFace
from retinaface.commons import preprocess, postprocess
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model



# Load models
model = load_model(r'./maskdetection_4000_32_100.h5')
rfModel = RetinaFace.build_model()



# Takes an CV2 image and outputs the same image with faces bounded and 
# an indication as to whether they are wearing masks

def detect_mask(image):
    # Find faces
    theFaces = detect_faces(image, model = rfModel)
    imageOutput = copy.deepcopy(image)

    for i, f in theFaces.items():
        conf = f["score"]
        print(i + " - Confidence: ",conf)

        # Cut out detected face
        fa = f["facial_area"]
        face = image[fa[1]: fa[3], fa[0]: fa[2]]

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

    return imageOutput



# RetinaFace function, with slight modification to take CV2 image object rather then image path
#
# Source: https://github.com/serengil/retinaface/blob/master/retinaface/RetinaFace.py
# Authors: Sefik Ilkin Serengil, Tobias Hermann

def detect_faces(img, threshold=0.9, model = None, allow_upscaling = True):
    if model is None:
        model = RetinaFace.build_model()

    #---------------------------

    nms_threshold = 0.4; decay4=0.5

    _feat_stride_fpn = [32, 16, 8]

    _anchors_fpn = {
        'stride32': np.array([[-248., -248.,  263.,  263.], [-120., -120.,  135.,  135.]], dtype=np.float32),
        'stride16': np.array([[-56., -56.,  71.,  71.], [-24., -24.,  39.,  39.]], dtype=np.float32),
        'stride8': np.array([[-8., -8., 23., 23.], [ 0.,  0., 15., 15.]], dtype=np.float32)
    }

    _num_anchors = {'stride32': 2, 'stride16': 2, 'stride8': 2}

    #---------------------------

    proposals_list = []
    scores_list = []
    landmarks_list = []
    im_tensor, im_info, im_scale = preprocess.preprocess_image(img, allow_upscaling)
    net_out = model(im_tensor)
    net_out = [elt.numpy() for elt in net_out]
    sym_idx = 0

    for _idx, s in enumerate(_feat_stride_fpn):
        _key = 'stride%s'%s
        scores = net_out[sym_idx]
        scores = scores[:, :, :, _num_anchors['stride%s'%s]:]

        bbox_deltas = net_out[sym_idx + 1]
        height, width = bbox_deltas.shape[1], bbox_deltas.shape[2]

        A = _num_anchors['stride%s'%s]
        K = height * width
        anchors_fpn = _anchors_fpn['stride%s'%s]
        anchors = postprocess.anchors_plane(height, width, s, anchors_fpn)
        anchors = anchors.reshape((K * A, 4))
        scores = scores.reshape((-1, 1))

        bbox_stds = [1.0, 1.0, 1.0, 1.0]
        bbox_deltas = bbox_deltas
        bbox_pred_len = bbox_deltas.shape[3]//A
        bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))
        bbox_deltas[:, 0::4] = bbox_deltas[:,0::4] * bbox_stds[0]
        bbox_deltas[:, 1::4] = bbox_deltas[:,1::4] * bbox_stds[1]
        bbox_deltas[:, 2::4] = bbox_deltas[:,2::4] * bbox_stds[2]
        bbox_deltas[:, 3::4] = bbox_deltas[:,3::4] * bbox_stds[3]
        proposals = postprocess.bbox_pred(anchors, bbox_deltas)

        proposals = postprocess.clip_boxes(proposals, im_info[:2])

        if s==4 and decay4<1.0:
            scores *= decay4

        scores_ravel = scores.ravel()
        order = np.where(scores_ravel>=threshold)[0]
        proposals = proposals[order, :]
        scores = scores[order]

        proposals[:, 0:4] /= im_scale
        proposals_list.append(proposals)
        scores_list.append(scores)

        landmark_deltas = net_out[sym_idx + 2]
        landmark_pred_len = landmark_deltas.shape[3]//A
        landmark_deltas = landmark_deltas.reshape((-1, 5, landmark_pred_len//5))
        landmarks = postprocess.landmark_pred(anchors, landmark_deltas)
        landmarks = landmarks[order, :]

        landmarks[:, :, 0:2] /= im_scale
        landmarks_list.append(landmarks)
        sym_idx += 3

    proposals = np.vstack(proposals_list)
    if proposals.shape[0]==0:
        landmarks = np.zeros( (0,5,2) )
        return np.zeros( (0,5) ), landmarks
    scores = np.vstack(scores_list)
    scores_ravel = scores.ravel()
    order = scores_ravel.argsort()[::-1]

    proposals = proposals[order, :]
    scores = scores[order]
    landmarks = np.vstack(landmarks_list)
    landmarks = landmarks[order].astype(np.float32, copy=False)

    pre_det = np.hstack((proposals[:,0:4], scores)).astype(np.float32, copy=False)

    #nms = cpu_nms_wrapper(nms_threshold)
    #keep = nms(pre_det)
    keep = postprocess.cpu_nms(pre_det, nms_threshold)

    det = np.hstack( (pre_det, proposals[:,4:]) )
    det = det[keep, :]
    landmarks = landmarks[keep]

    resp = {}
    for idx, face in enumerate(det):

        label = 'face_'+str(idx+1)
        resp[label] = {}
        resp[label]["score"] = face[4]

        resp[label]["facial_area"] = list(face[0:4].astype(int))

        resp[label]["landmarks"] = {}
        resp[label]["landmarks"]["right_eye"] = list(landmarks[idx][0])
        resp[label]["landmarks"]["left_eye"] = list(landmarks[idx][1])
        resp[label]["landmarks"]["nose"] = list(landmarks[idx][2])
        resp[label]["landmarks"]["mouth_right"] = list(landmarks[idx][3])
        resp[label]["landmarks"]["mouth_left"] = list(landmarks[idx][4])

    return resp