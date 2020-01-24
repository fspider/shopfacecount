import time
import numpy as np
import face_recognition
from collections import deque
import argparse
import configparser
import cv2 as cv
import datetime
from yolo_utils import  estimate_age_gender_emotion1
import warnings
import csv
from pymongo import MongoClient
import datetime, time

def get_realBox(boxs, cp):
    cx, cy = cp
    index = -1
    M = -1
    for i in range(len(boxs)):
        x, y, w, h = boxs[i]
        x0 = x + w / 2
        y0 = y + h / 2
        d = np.sqrt((x0 - cx) ** 2 + (cy - y0) ** 2)
        if(M == -1):
            M = d
            index = i
        else:
            if(M > d):
                M = d
                index = i
    return index
def get_json(feed):
    data = {}
    fields = ['id', 'x1', 'x2', 'y1', 'y2', 'classification', 'accuracy', 'speed', 'actions', 'express',
              'gender', 'age', 'LPR']
    for i in range(len(feed)):
        data[fields[i]] = feed[i]
    return data

map_track_id = {}
cnt_track_id = 0

def get_track_id(track_id) :
    global cnt_track_id
    global map_track_id

    if (track_id in map_track_id) :
        return map_track_id[track_id]
    else:
        map_track_id[track_id] = cnt_track_id
        cnt_track_id = cnt_track_id + 1
        return cnt_track_id - 1

def drawlineSeg(frame, x1, y1, x2, y2, col, wid) :
    midx1 = (x1 * 3 + x2) / 4
    midx2 = (x1 + x2 * 3) / 4
    midy1 = (y1 * 3 + y2) / 4
    midy2 = (y1 + y2 * 3) / 4
    cv.line(frame, (x1, y1), (int(midx1), int(midy1)), col, wid)
    cv.line(frame, (int(midx2), int(midy2)), (x2, y2), col, wid)

def drawRectangle(frame, st, ed, col, wid):
    [x1, y1] = st
    [x2, y2] = ed
    drawlineSeg(frame, x1, y1, x1, y2, col, wid)
    drawlineSeg(frame, x1, y1, x2, y1, col, wid)
    drawlineSeg(frame, x2, y2, x1, y2, col, wid)
    drawlineSeg(frame, x2, y2, x2, y1, col, wid)



# This is the main entry Function
if __name__ == '__main__':
    faces = []
    ids = []
    nfaces = 0


    warnings.filterwarnings('ignore')
    config = configparser.ConfigParser()
    config.read('config.ini')
    videoUrl = 'D:/e.mp4' #getVideoUrl(config)
    csv_file = videoUrl + '_out.csv'

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-path',
        type=str,
        default= config['DEFAULT']['model-path'],
        help='The directory where the model weights and \
            configuration files are.')

    parser.add_argument('-w', '--weights',
        type=str,
        default=config['DEFAULT']['weights'],
        help='Path to the file which contains the weights \
            for YOLOv3.')

    parser.add_argument('-cfg', '--config',
        type=str,
        default=config['DEFAULT']['config'],
            help='Path to the configuration file for the YOLOv3 model.')

    parser.add_argument('-l', '--labels',
        type=str,
        default='./models/coco_classes.txt',
        help='Path to the file having the \
            labels in a new-line seperated way.')

    parser.add_argument('-c', '--confidence',
        type=float,
        default=config['DEFAULT']['confidence'],
        help='The model will reject boundaries which has a \
            probabiity less than the confidence value. \
            default: 0.5')

    parser.add_argument('-th', '--threshold',
        type=float,
        default=config['DEFAULT']['threshold'],
        help='The threshold to use when applying the \
            Non-Max Suppresion')

    parser.add_argument('-t', '--show-time',
        type=bool,
        default=config['DEFAULT']['show-time'],
        help='Show the time taken to infer each image.')



    FLAGS, unparsed = parser.parse_known_args()


    writeVideo_flag = True
    # Openc VideoCapture
    try:
        cap = cv.VideoCapture(videoUrl)
        fps = cap.get(cv.CAP_PROP_FPS)
        print(fps)

    except:
        raise Exception('Video cannot be loaded!\n\
                            Please check the path provided!')

    finally:
        count = 0
        HH, WW = int(cap.get(3)), int(cap.get(4))
        prev_time = datetime.datetime.now()
        if writeVideo_flag:
            # Define the codec and create VideoWriter object
            fourcc = cv.VideoWriter_fourcc(*'MJPG')
            out = cv.VideoWriter(videoUrl + '_output.avi', fourcc, 15, (HH, WW))
        frame_num = 0

        while True:
            grabbed, image = cap.read()

            # Checking if the complete video is read
            if not grabbed:
                break
            frame_num += 1
            if(frame_num % 5 !=0):
                continue
            start = time.time()

            frame = image.copy()
            age_pre, gen_pre, emo_pre, boxs = estimate_age_gender_emotion1(frame)
            boxes = [ [y, x+w, y+h, x,] for x, y, w, h in boxs]
            print(boxes)
            dfaces = face_recognition.face_encodings(frame, boxes)

            if(boxs is not None):

                i = -1
                for bbox in boxes:
                    i = i + 1
                    bbox = boxs[i]
                    x1 = int(bbox[0])
                    y1 = int(bbox[1])
                    x2 = x1 + int(bbox[2])
                    y2 = y1 + int(bbox[3])

                    img = frame[y1:y2, x1:x2].copy()
#                    cv.imshow('face', img)
#                    face = face_recognition.face_encodings(img)
                    face = dfaces[i]
                    results = face_recognition.compare_faces(faces, face)
                    flag = False

                    k = 0
                    for result in results:
                        if result == 1:
                            flag = True
                            break
                        k = k + 1

                    if flag == True:
                        drawRectangle(frame, (x1, y1), (x2, y2), (106, 206, 15), 2)
                        cv.putText(frame, str(ids[k]), (x1, int(y1 - 10)), 0, 5e-3 * 50, (255, 0, 255), 1)
                        print(k)
                    else:
                        print("OKOKOK")
                        if nfaces < 20:
                            faces.append(face)
                            ids.append(nfaces+1)
                        else:
                            faces[nfaces % 20] = face
                            ids[nfaces % 20] = nfaces + 1
                        nfaces = nfaces + 1

                        drawRectangle(frame, (x1, y1), (x2, y2), (106, 15, 206), 2)
                        cv.putText(frame, str(nfaces), (x1, int(y1 - 10)), 0, 5e-3 * 50, (255, 0, 255), 1)
                    

                    gen_v = 'F'
                    if(gen_pre[i][0] < 0.5):
                        gen_v = 'M'
                    label_pre = "{}, {}, {}".format(int(age_pre[i]), gen_v, emo_pre[i])
                    cv.putText(frame, label_pre, (x1, int(y1 - 20)), 0, 5e-3 * 80,
                                (0, 0, 255), 1)


            cv.putText(frame, str(nfaces), (50, 50)  , 0, 5e-3 * 200, (255, 0, 0), 3)
            if writeVideo_flag:
                # save a frame
                out.write(frame)
            # Display Video
            cv.imshow('video', frame)
            end = time.time()
            print('fps:', 1 / (end - start))
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        if writeVideo_flag:
            out.release()
        cap.release()
