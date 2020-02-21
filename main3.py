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
from ConfigRgn import  ConfigRgn
from Face_Entity import Face_collect

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

def draw_regon(image, nodes_b, nodes_c):
    cv.polylines(image, nodes_b, True, (255, 0, 0), 3)
    cv.polylines(image, nodes_c, True, (0, 0, 255), 3)


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


def create_table2(cur, conn, cameraid):
    # This command deletes table if already exists
    # query = """DROP TABLE recognition2_""" + str(cameraid)
    # cur.execute(query)
    # conn.commit()
    # print("delete success")
    try:


        query =  """
            CREATE TABLE recognition2_""" + str(cameraid) + """ (
                id integer PRIMARY KEY,
                camera_id integer,
                frame_no integer,
                face_id integer,
                age integer,
                gender character,
                emotion character varying[50],
                access_time character varying[50],
                start_time timestamp without time zone,
                stop_time timestamp without time zone
            )
            """
        cur.execute(query)
        conn.commit()
    except:
        cur.execute("ROLLBACK")
        conn.commit()
        print (' TABLE recognition2_', str(cameraid), ' already EXISTS')

def draw_region(frame, nds, color):
    cv.fillPoly(frame, nds, color)

def reconnet_DB(config):
    try:
        # Connect to Postgresql Database
        print('Reconnecting to the PostgreSQL database...')
        conn = psycopg2.connect(host=config['postgresql']['host'],
                                database=config['postgresql']['database'],
                                user=config['postgresql']['user'],
                                password=config['postgresql']['password'],
                                port=config['postgresql']['port'])
        cur = conn.cursor()    
        return cur
    except:
        print('Reconnecting to the PostgreSQL database Failed...')
        return 0

# This is the main entry Function
if __name__ == '__main__':

    # if this values is set, output result video.
    writeVideo_flag = False
    # if this value is set, output data on Postgresql Database
    writeDb_flag = True

    warnings.filterwarnings('ignore')
    config = configparser.ConfigParser()
    config.read('config.ini')
    videoUrl = 0 #getVideoUrl(config)
    #videoUrl = config['INPUT']['VideoURL']
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
        create_table2(cur, conn, cameraid)

        # Get the last id , so that we can continue counting
        query = """select max(id) from recognition2_""" + str(cameraid)
        cur.execute(query)
        result = cur.fetchall()
        nfaces = result[0][0]
        if not nfaces:
            nfaces = 0

    # Open VideoCapture
    try:
        cap = cv.VideoCapture(videoUrl)
        fps = cap.get(cv.CAP_PROP_FPS)
        #print(fps)

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

        grabbed, image = cap.read()
        if not grabbed:
            print('ERROR: failed to open video.....')
            sys.exit(0)

        roi_flag = int(config['INPUT']['DRAW_ROI'])
        roi_filename = config['INPUT']['ROI_FILENAME']
        cfgRgn = ConfigRgn(image, roi_filename)

        if roi_flag == 1:
            nodes_b, mask_b, nodes_c, mask_c  = cfgRgn.setRegion()
        else: # loading from roi_filename
            result, tpl = cfgRgn.load_roi()
            if not result:
                print('cannot open config file')
                sys.exit(-1)
            nodes_b, mask_b, nodes_c, mask_c = tpl

        min_face_size = int(config['INPUT']['min_face_size'])

        face_manager = Face_collect(mask_b,mask_c,  nfaces, min_access_time=5, min_leave_time= 5, min_face = min_face_size)

        while True:
            start = time.time()
            cap.set(cv.CAP_PROP_FOURCC,cv.VideoWriter_fourcc('M','J','P','G'))
            grabbed, image = cap.read()

            # Checking if the complete video is read
            if(not grabbed) or (image is None):
                break
            frame_num += 1
            #if frame_num % 5 !=0:
            #    continue

            remove_list = face_manager.Update(image)

            if writeDb_flag is True:
                # Get last id
                try:
                    query = """select max(id) from recognition2_""" + str(cameraid)
                    cur.execute(query)
                    result = cur.fetchall()
                    nid = result[0][0]
                    if not nid:
                        nid = 1
                    else:
                        nid = nid + 1
                    if remove_list is not None:
                        for one in remove_list:
                            # insert record to database
                            insert_query = """ INSERT INTO recognition2_"""+str(cameraid)+""" (id, camera_id, frame_no, face_id, age, gender, emotion, access_time, start_time, stop_time)\
                                                                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
                            delta = one.get_access_time_str()
                            record_to_insert = (nid, cameraid, frame_num, one.id, int(one.age), one.gen, '{'+ one.emotion +'}', '{' + delta + '}', one.start, one.stop)
                            cur.execute(insert_query, record_to_insert)
                            conn.commit()

                            inserted_record = 'inserted item: {} {} {} {} {} {} {} {} {} {}'.format(nid, cameraid, frame_num, one.id, one.age, one.gen, one.emotion, delta, one.start, one.stop)
                            print(inserted_record)
                            nid += 1
                except Exception as error:
                    print('Exception : ' + repr(error))
                    cur = reconnet_DB(config)
                    pass

            # If one day pass, init today_cnt
            now_date = datetime.datetime.now().date()
            if prev_date != now_date:
                today_cnt = 0
            prev_date = now_date

            # calculate fps
            end = time.time()
            #print('fps:', 1 / (end - start))

            if writeVideo_flag:
                # save a frame
                out.write(image)

            # Display Video
            draw_regon(image, nodes_b, nodes_c)

            cv.putText(image, str(face_manager.today_face()), (50, 50), 0, 5e-3 * 200, (255, 0, 0), 3)
            # draw today number of faces
            cv.putText(image, str(face_manager.total_face()), (50, 100), 0, 5e-3 * 200, (255, 0, 0), 3)
            cv.imshow('video', image)
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

