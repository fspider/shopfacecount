import numpy as np
import cv2 as cv
from keras.models import load_model
from pathlib import Path
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file
from mtcnn.mtcnn import MTCNN

# age, gender, emotion estimation
# face detector
detector = MTCNN()
# emotion estimator
train_model = "ResNet"  # (Inception-v3, Inception-ResNet-v2): Inception,  (ResNet-50): ResNet

# Size of the images
if train_model == "Inception":
    img_width, img_height = 139, 139
elif train_model == "ResNet":
    img_width, img_height = 197, 197

emotions = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']

# Reinstantiate the fine-tuned model (Also compiling the model using the saved training configuration (unless the model was never compiled))
emo_model = load_model('./trained_models/ResNet-50.h5')
# age and gender
weight_file = None
margin = 0.4
img_size = 64
model = WideResNet(img_size, depth=16, k=8)()
model.load_weights('./trained_models/weights.28-3.73.hdf5')


def preprocess_input(image):
    image = cv.resize(image, (img_width, img_height))
    ret = np.empty((img_height, img_width, 3))
    ret[:, :, 0] = image
    ret[:, :, 1] = image
    ret[:, :, 2] = image
    x = np.expand_dims(ret, axis=0)  # (1, XXX, XXX, 3)

    if train_model == "Inception":
        x /= 127.5
        x -= 1.
        return x
    elif train_model == "ResNet":
        x -= 128.8006  # np.mean(train_dataset)
        x /= 64.6497  # np.std(train_dataset)

    return x


def predict(emotion):
    # Generates output predictions for the input samples
    # x:    the input data, as a Numpy array (None, None, None, 3)
    prediction = emo_model.predict(emotion)

    return prediction
def estimate_age_gender_emotion(img):
    img_h, img_w, _ = np.shape(img)
    try:
        gray_frame = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Conversion of the image to the grayscale
        detected = detector.detect_faces(img)
        # detected = detector(input_img, 1)
        faces = np.empty((1, img_size, img_size, 3))
        if len(detected) > 0:
            face = detected[0]
            (x, y, w, h) = face['box']
            if (x < 0):
                x = 0
            if (y < 0):
                y = 0
            x1, y1, x2, y2 = x, y, x + w + 1, y + h + 1
            xw1 = max(int(x1 - margin * w), 0)
            yw1 = max(int(y1 - margin * h), 0)
            xw2 = min(int(x2 + margin * w), img_w - 1)
            yw2 = min(int(y2 + margin * h), img_h - 1)
            faces[0, :, :, :] = cv.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
            ROI_gray = gray_frame[y:y + h, x:x + w]  # Extraction of the region of interest (face) from the frame
            emotion = preprocess_input(ROI_gray)
            prediction = predict(emotion)
            emo_pre = emotions[np.argmax(prediction)]
            # predict ages and genders of the detected faces
            results = model.predict(faces)
            predicted_genders = results[0]
            if(predicted_genders[0][0] < 0.5):
                gender_pre = 'M'
            else:
                gender_pre = 'F'
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()
            return int(predicted_ages[0]), gender_pre, emo_pre
    except:
        pass
    return None, None, None

def estimate_age_gender_emotion1(img):
    img_h, img_w, _ = np.shape(img)
    emotion_pred = []
    ages_pre = []
    predicted_genders = []
    boxs = []
    gray_frame = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Conversion of the image to the grayscale
    detected = detector.detect_faces(img)
    faces = np.empty((len(detected), img_size, img_size, 3))
    if len(detected) > 0:
        for i, face in enumerate(detected):
            (x, y, w, h) = face['box']
            if (x < 0):
                x = 0
            if (y < 0):
                y = 0
            x1, y1, x2, y2 = x, y, x + w + 1, y + h + 1
            # x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            xw1 = max(int(x1 - margin * w), 0)
            yw1 = max(int(y1 - margin * h), 0)
            xw2 = min(int(x2 + margin * w), img_w - 1)
            yw2 = min(int(y2 + margin * h), img_h - 1)
            boxs.append([x, y, w, h])
            faces[i, :, :, :] = cv.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
            ROI_gray = gray_frame[y:y + h, x:x + w]  # Extraction of the region of interest (face) from the frame
            emotion = preprocess_input(ROI_gray)
            prediction = predict(emotion)
            emo_pre = emotions[np.argmax(prediction)]
            emotion_pred.append(emo_pre)
        # predict ages and genders of the detected faces
        results = model.predict(faces)
        predicted_genders = results[0]
        ages_v = np.arange(0, 101).reshape(101, 1)
        ages_pre = results[1].dot(ages_v).flatten()
    return ages_pre, predicted_genders, emotion_pred, boxs


# def draw_labels_and_boxes(img, boxes, confidences, classids, idxs, colors, labels):
#     # If there are any detections
#     if len(idxs) > 0:
#         for i in idxs.flatten():
#             # Get the bounding box coordinates
#             x, y = boxes[i][0], boxes[i][1]
#             w, h = boxes[i][2], boxes[i][3]
#
#             # Get the unique color for this class
#             color = [int(c) for c in colors[classids[i]]]
#
#             # Draw the bounding box rectangle and label on the image
#             cv.rectangle(img, (x, y), (x+w, y+h), color, 2)
#             text = "{}: {:4f}".format(labels[classids[i]], confidences[i])
#             cv.putText(img, text, (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#
#     return img


def generate_boxes_confidences_classids(outs, height, width, tconf):
    boxes = []
    confidences = []
    classids = []

    for out in outs:
        for detection in out:
            # Get the scores, classid, and the confidence of the prediction
            scores = detection[5:]
            classid = np.argmax(scores)
            confidence = scores[classid]
            if(classid != 0 and classid != 2 and classid != 3 and classid != 5 and classid != 7):
                continue
            # Consider only the predictions that are above a certain confidence level
            if confidence > tconf:
                # TODO Check detection
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, bwidth, bheight = box.astype('int')
                # Using the center x, y coordinates to derive the top
                # and the left corner of the bounding box
                x = int(centerX - (bwidth / 2))
                y = int(centerY - (bheight / 2))
                # Append to list
                boxes.append([x, y, int(bwidth), int(bheight)])
                confidences.append(float(confidence))
                classids.append(classid)
    return boxes, confidences, classids

def infer_image(net, layer_names, img, FLAGS):
    
    # Contructing a blob from the input image
    blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
                    swapRB=True, crop=False)
    # Perform a forward pass of the YOLO object detector
    net.setInput(blob)
    # Getting the outputs from the output layers
    outs = net.forward(layer_names)
    height, width = img.shape[:2]
    # Generate the boxes, confidences, and classIDs
    boxes, confidences, classids = generate_boxes_confidences_classids(outs, height, width, FLAGS.confidence)

    # Apply Non-Maxima Suppression to suppress overlapping bounding boxes
    idxs = cv.dnn.NMSBoxes(boxes, confidences, FLAGS.confidence, FLAGS.threshold)
    if boxes is None or confidences is None or idxs is None or classids is None:
        return img, None, None, None
    else:
        return img, boxes, confidences, classids