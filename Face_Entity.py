import numpy as np
import cv2 as cv
from yolo_utils import  estimate_age_gender_emotion1
import face_recognition
import datetime, time

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



class Face_Entity:
    def __init__(self, id, face, age, gen, emotion):
        self.id = id        # face id
        self.face = face    # face embedding information
        self.age = age      # age
        self.gen = gen      # gender
        self.emotion = emotion  # emotion
        self.start = datetime.datetime.now()    # start timestamp to access
        self.stop = datetime.datetime.now()     # end timestamp to access
        self.access_time = '00:00:00'

    def update_face(self, face): # exchange the face's embedding and update the last timestamp
        self.face = face
        self.stop = datetime.datetime.now()
        elasped = (self.stop - self.start).seconds
        H = elasped // 3600
        M = (elasped % 3600) // 60
        S = elasped % 60
        self.access_time = '{:02}:{:02}:{:02}'.format(H, M, S)
        #print('id = {}, access_time = {}'.format(id, self.access_time))

    def get_access_time_str(self):# get total access time as a format string
        return self.access_time


    def get_access_time(self):# get total access time
        return int((self.stop - self.start).seconds)

    def get_losttime_second(self): # get the time from last detection timestamp as a second unit
        return (datetime.datetime.now()- self.stop).seconds


# management class of Face_Entity array
class Face_collect:
    def __init__( self,  mask_b, mask_c, nfaces, min_access_time = 5,  min_leave_time = 5, min_face = 0):

        self.faces_collects = []    #  list of face_entity
        self.mask_b = mask_b.copy() # mask of buyer's region
        self.mask_c = mask_c.copy() # mask of cashier's region
        self.faces_cashier = []     # faces embedding of cashiers
        self.min_face = min_face    # min size of we need to detect/track
        self.faces = []
        self.cur_id = 1             # start face_id
        self.nFaces = 0
        self.nTotalCunt = nfaces
        # if one person accss the chier more than this time, determine this person as a buyer
        self.min_access_time = min_access_time
        # if some face don't occur more than this time in mask region, determin this person go away
        self.min_leave_time = min_leave_time
        self.prev_date = datetime.datetime.now().date()

    def today_face(self):
        return self.nFaces

    # return the number of current tracked faces
    def face_count(self):
        return len(self.faces_collects)

    def total_face(self):
        return self.nTotalCunt

    # return the new id to add to list
    def get_new_id(self):
        return self.cur_id

    # add the new face to list
    def add_new_face(self, face, age, gen, emotion):
        id = self.get_new_id()
        face_entity = Face_Entity(id, face, age, gen, emotion)
        self.faces_collects.append(face_entity)
        self.cur_id += 1
        self.nFaces += 1
        self.nTotalCunt += 1


    # check this person is out of buyer's regon
    def is_outOfBuyerRgn(self, x1, y1, x2, y2):
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        if self.mask_b[y1][x1] == 0:
            return True
        return False

    # this one face occur in cashier's region
    def is_cashier(self, x1, y1, x2, y2):
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        if self.mask_c[y1][x1] == 0:
            return False
        return True

    # check the size of face to exceed the min_size
    def is_small_face(self, bbox):
        w, h = bbox[2], bbox[3]
        if w < self.min_face or h < self.min_face:
            return True
        return False

    # find all lost faces and remove it from face_entity list
    def get_lost_faces(self):
        remove_list = []
        face_count = self.face_count()
        for cnt in range(face_count - 1, -1, -1):
            face = self.faces_collects[cnt]
            # get access time
            access_time = face.get_access_time()

            # access time too less
            if access_time < self.min_access_time:
                continue
            # check lost time
            elapsed = face.get_losttime_second()
            # print('id: {}, lost time = {}'.format(face.id, elapsed))

            # add to new list and remove it from current stack
            if elapsed >= self.min_leave_time:
                remove_list.append(face)
                del self.faces_collects[cnt]
        return remove_list

    # update the current face tracks from this image
    def Update(self, frame):
        # detect face:
        age_pre, gen_pre, emo_pre, boxs = estimate_age_gender_emotion1(frame)
        boxes = [[y, x + w, y + h, x, ] for x, y, w, h in boxs]
        #print(boxes)
        # Encode faces for face compare
        dfaces = face_recognition.face_encodings(frame, boxes)
        if (boxs is not None):
            i = -1

            for bbox in boxes:
                i = i + 1
                bbox = boxs[i]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = x1 + int(bbox[2])
                y2 = y1 + int(bbox[3])

                face = dfaces[i]

                # check size
                if self.is_small_face(bbox):
                    continue
                # check cashier
                if self.is_cashier(x1, y1, x2, y2):
                    result = face_recognition.compare_faces(self.faces_cashier, face)
                    if len(result)== 0 or max(result) == 0: # new cashier
                        print('find cashier face in cashier region' )
                        if(len(self.faces_cashier) < 5) :
                            self.faces_cashier.append(face)
                        drawRectangle(frame, (x1, y1), (x2, y2), (106, 15, 206), 2)
                        cv.putText(frame, 'cashier', (x1, int(y1 - 10)), 0, 5e-3 * 100, (255, 0, 0), 2)
                    continue

                if self.is_outOfBuyerRgn(x1, y1, x2, y2): # out of buyer region
                    continue

                # compare this new face with cashier's faces
                results = face_recognition.compare_faces(self.faces_cashier, face)
                bFind = False
                for result in results:
                    if result == 1:
                        bFind = True
                        break
                if bFind: # find cashier's face in buyer region
                    print('find cashier in buyer region')
                    drawRectangle(frame, (x1, y1), (x2, y2), (106, 15, 206), 2)
                    cv.putText(frame, 'cashier', (x1, int(y1 - 10)), 0, 5e-3 * 100, (255, 0, 0), 2)
                    continue

                # compare face with buyer's region
                faces = [ face_entity.face for face_entity in self.faces_collects]
                results = face_recognition.compare_faces(faces, face)
                flag = False

                # Find if there is matching Face
                k = 0
                for result in results:
                    if result == 1:
                        flag = True
                        break
                    k = k + 1

                if flag == True: # already detected person
                    self.faces_collects[k].update_face(face)
                    cur_face = self.faces_collects[k]

                else: # new person
                    gen_v = 'F'
                    if (gen_pre[i][0] < 0.5):
                        gen_v = 'M'

                    self.add_new_face(face, int(age_pre[i]), gen_v, emo_pre[i])
                    cur_face = self.faces_collects[self.face_count() - 1]


                drawRectangle(frame, (x1, y1), (x2, y2), (106, 15, 206), 2)
                # draw face_id for each face
                cv.putText(frame, str(cur_face.id), (x1, int(y1 - 10)), 0, 5e-3 * 100, (255, 0, 0),2)

                label_pre = "{}, {}, {}".format(cur_face.age, cur_face.gen, cur_face.emotion)
                cv.putText(frame, label_pre, (x1, int(y1 - 20)), 0, 5e-3 * 100,
                           (0, 0, 255), 2)

        now_date = datetime.datetime.now().date()
        if now_date != self.prev_date:
            self.nFaces = 0
            self.prev_date = now_date

        remove_list = self.get_lost_faces()
        return remove_list


