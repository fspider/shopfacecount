import time
import numpy as np
import face_recognition
from collections import deque
import argparse
import configparser
import cv2 as cv
from yolo_utils import  estimate_age_gender_emotion1
import warnings
import csv
import datetime, time
import psycopg2
import sys

# Find the nearest box in boxs from cp point
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
# get json object from feed object
def get_json(feed):
    data = {}
    fields = ['id', 'x1', 'x2', 'y1', 'y2', 'classification', 'accuracy', 'speed', 'actions', 'express',
              'gender', 'age', 'LPR']
    for i in range(len(feed)):
        data[fields[i]] = feed[i]
    return data

# Map the tracking id from id sequencing
map_track_id = {}
cnt_track_id = 0

# Get the sequence id from track_id
def get_track_id(track_id) :
    global cnt_track_id
    global map_track_id

    if (track_id in map_track_id) :
        return map_track_id[track_id]
    else:
        map_track_id[track_id] = cnt_track_id
        cnt_track_id = cnt_track_id + 1
        return cnt_track_id - 1

# drawlinesegmet typed like - -
def drawlineSeg(frame, x1, y1, x2, y2, col, wid) :
    midx1 = (x1 * 3 + x2) / 4
    midx2 = (x1 + x2 * 3) / 4
    midy1 = (y1 * 3 + y2) / 4
    midy2 = (y1 + y2 * 3) / 4
    cv.line(frame, (x1, y1), (int(midx1), int(midy1)), col, wid)
    cv.line(frame, (int(midx2), int(midy2)), (x2, y2), col, wid)

# draw rectangle for face
def drawRectangle(frame, st, ed, col, wid):
    [x1, y1] = st
    [x2, y2] = ed
    drawlineSeg(frame, x1, y1, x1, y2, col, wid)
    drawlineSeg(frame, x1, y1, x2, y1, col, wid)
    drawlineSeg(frame, x2, y2, x1, y2, col, wid)
    drawlineSeg(frame, x2, y2, x2, y1, col, wid)

# if there is no table with cameraid, create one table named as recognition_{camera-id}
def create_table(cur, conn, cameraid):
    # This command deletes table if already exists
    # query =  """DROP TABLE recognition_""" + str(cameraid)
    # cur.execute(query)
    # conn.commit()
    try:
        query =  """
            CREATE TABLE recognition_""" + str(cameraid) + """ (
                id integer PRIMARY KEY,
                camera_id integer,
                frame_no integer,
                face_id integer,
                age integer,
                gender character,
                emotion character varying[50],
                in_time timestamp without time zone
            )
            """
        cur.execute(query)
        conn.commit()
    except:
        cur.execute("ROLLBACK")
        conn.commit()
        print (' TABLE recognition_', str(cameraid), ' already EXISTS')


# This is the main entry Function
if __name__ == '__main__':
    faces = []
    ids = []
    nfaces = 0
    cnt = 0
    today_cnt = 0

    # if this values is set, output result video.
    writeVideo_flag = False
    # if this value is set, output data on Postgresql Database
    writeDb_flag = True

    warnings.filterwarnings('ignore')
    config = configparser.ConfigParser()
    config.read('config.ini')
    videoUrl = 0 #getVideoUrl(config)
    # videoUrl = config['INPUT']['VideoURL']
    cameraid = config['DEFAULT']['camera-id']

    if writeDb_flag is True:
        # Connect to Postgresql Database
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(host=config['postgresql']['host'],
                                database=config['postgresql']['database'],
                                user=config['postgresql']['user'],
                                password=config['postgresql']['password'],
                                port=config['postgresql']['port'])
        cur = conn.cursor()

        # Check and Create table for this camera id
        create_table(cur, conn, cameraid)

        # Get the last face_id , so that we can continue counting
        query = """select max(face_id) from recognition_""" + str(cameraid)
        cur.execute(query)
        result = cur.fetchall()
        nfaces = result[0][0]
        if not nfaces:
            nfaces = 0


    # Open VideoCapture
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
        if writeVideo_flag:
            # Define the codec and create VideoWriter object
            fourcc = cv.VideoWriter_fourcc(*'MJPG')
            out = cv.VideoWriter(str(videoUrl) + '_output.avi', fourcc, 15, (HH, WW))
        frame_num = 0
        prev_date = datetime.datetime.now().date()

        while True:
            cap.set(cv.CAP_PROP_FOURCC,cv.VideoWriter_fourcc('M','J','P','G'))
            grabbed, image = cap.read()

            # Checking if the complete video is read
            if not grabbed:
                break
            frame_num += 1
            # if(frame_num % 5 !=0):
            #     continue
            start = time.time()

            frame = image.copy()
            # Get age, gender, emotion, boxes from frame iamge
            age_pre, gen_pre, emo_pre, boxs = estimate_age_gender_emotion1(frame)
            boxes = [ [y, x+w, y+h, x,] for x, y, w, h in boxs]
            print(boxes)
            # Encode faces for face compare
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

                    # Get only Face Image from Frame
                    img = frame[y1:y2, x1:x2].copy()
#                    cv.imshow('face', img)
#                    face = face_recognition.face_encodings(img)
                    face = dfaces[i]
                    # compare face with last faces
                    results = face_recognition.compare_faces(faces, face)
                    flag = False

                    # Find if there is matching Face
                    k = 0
                    for result in results:
                        if result == 1:
                            flag = True
                            break
                        k = k + 1

                    # If there is matching face, draw green rectangle
                    if flag == True:
                        face_id = ids[k]
                        drawRectangle(frame, (x1, y1), (x2, y2), (106, 206, 15), 2)
                        print(k)
                    # else draw red rectangle
                    else:
                        if cnt < 20:
                            faces.append(face)
                            ids.append(nfaces+1)
                        else:
                            faces[cnt % 20] = face
                            ids[cnt % 20] = nfaces + 1
                        # If new face, count + 1
                        nfaces = nfaces + 1
                        cnt = cnt + 1
                        today_cnt = today_cnt + 1
                        face_id = nfaces
                        drawRectangle(frame, (x1, y1), (x2, y2), (106, 15, 206), 2)

                    # draw face_id for each face
                    cv.putText(frame, str(face_id), (x1, int(y1 - 10)), 0, 5e-3 * 50, (255, 0, 255), 1)
                    

                    gen_v = 'F'
                    if(gen_pre[i][0] < 0.5):
                        gen_v = 'M'
                    label_pre = "{}, {}, {}".format(int(age_pre[i]), gen_v, emo_pre[i])
                    #draw age text
                    cv.putText(frame, label_pre, (x1, int(y1 - 20)), 0, 5e-3 * 80,
                                (0, 0, 255), 1)

                    if writeDb_flag is True:
                        # Get last id
                        query = """select max(id) from recognition_""" + str(cameraid)
                        cur.execute(query)
                        result = cur.fetchall()
                        nid = result[0][0]
                        if not nid:
                            nid = 1
                        else:
                            nid = nid + 1

                        # insert record to database
                        dt = datetime.datetime.now()
                        insert_query = """ INSERT INTO recognition_"""+str(cameraid)+""" (id, camera_id, frame_no, face_id, age, gender, emotion, in_time)\
                                                                     VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"""
                        record_to_insert = (nid, cameraid, frame_num, face_id, int(age_pre[i]), gen_v, '{'+emo_pre[i]+'}', dt)
                        cur.execute(insert_query, record_to_insert)
                        conn.commit()

            # If one day pass, init today_cnt
            now_date = datetime.datetime.now().date()
            if prev_date != now_date:
                today_cnt = 0
            prev_date = now_date

            #draw total number of faces
            cv.putText(frame, str(nfaces), (50, 50)  , 0, 5e-3 * 200, (255, 0, 0), 3)
            #draw today number of faces
            cv.putText(frame, str(today_cnt), (50, 100)  , 0, 5e-3 * 200, (255, 0, 0), 3)

            # calculate fps
            end = time.time()
            print('fps:', 1 / (end - start))

            if writeVideo_flag:
                # save a frame
                out.write(frame)
            # Display Video
            cv.imshow('video', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        if writeVideo_flag:
            out.release()
        cap.release()
    if writeDb_flag is True:
        cur.close()
        if conn is not None:
            conn.close()
            print('Database connection closed.')

