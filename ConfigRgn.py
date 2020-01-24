import cv2
import numpy as np
import threading
import os

# how to draw for buyer/cashier region

# if you remove the last point, please press 'r'
# if you complete a first region(this is a buyer's region), please press 'q'.
# And then automatically convert into state , which can draw cashier's region
# if you complete a cashier region, please press 'q'


# mouse callback function
def on_mouse(event, x, y, flag, param):
    param.mouse_cb(event, x, y, flag)



class ConfigRgn:
    def __init__(self, img, roi_file):
        self.img = img.copy() # confign initial image
        self.winname = "region config window"
        self.keyPoint = [] # temp key point
        self.roi_fn = roi_file # roi file name

        self.keyPoint_buyer = [] # keypoint array of buyer regoin
        self.keyPoint_cashier = [] # cashier region
        self.CLR_CIRCLE= (0,0,255) # color of circle
        self.CLR_LINE = (255, 0, 0) # line color of buyer
        self.CLR_LINE_S = (0,0,255) # line color of cashier

        self.bState = False # if false, drawing of buyer, else, drawing of cashier
        self.lock = threading.Lock()
        self.radius = 4 # circle radius when drawing it on image
        self.thick = 2

    # set the region of buyer/cashier using inital image
    def setRegion(self):
        cv2.namedWindow(self.winname)
        cv2.setMouseCallback(self.winname, on_mouse, self)
        self.bInit = False
        cv2.imshow(self.winname, self.img)
        while True:
            self.draw_region()
            key = cv2.waitKey(40)

            if key == ord('r'): # remove last key point
                key_len = len(self.keyPoint)
                if key_len <= 0:
                    continue

                self.lock.acquire()
                try:
                    del self.keyPoint[key_len - 1]
                finally:
                    self.lock.release()
                continue

            elif key == ord('q'): # complete of setting region
                self.lock.acquire()
                if not self.bState: # complete to draw buyer region
                    try:
                        self.keyPoint_buyer = self.keyPoint.copy()
                        self.keyPoint = []
                        self.bState = True
                    finally:
                        self.lock.release()
                    continue
                else: # complete to draw the cashier region
                    try:
                        self.keyPoint_cashier = self.keyPoint.copy()
                    finally:
                        self.lock.release()
                    break

        # make mask from keypoint array
        h, w, _ = self.img.shape
        mask_buyer = np.zeros((h, w), np.uint8)
        nds_buyer = np.array([self.keyPoint_buyer])
        cv2.fillPoly(mask_buyer, nds_buyer, 255)

        # cashier mask
        mask_cashier = np.zeros((h, w), np.uint8)
        nds_cashier = np.array([self.keyPoint_cashier])
        cv2.fillPoly(mask_cashier, nds_cashier, 255)

        # cv2.imwrite('buyer.jpg', mask_buyer)
        # cv2.imwrite('cashier.jpg', mask_cashier)
        cv2.destroyWindow(self.winname)
        self.save_roi()

        return nds_buyer, mask_buyer, nds_cashier, mask_cashier

    # save current keypoint as a txt file
    def save_roi(self):
        with open(self.roi_fn, 'w') as fp:
            str_buy = ''
            str_cas = ''
            for pt in self.keyPoint_buyer:
                str_buy = str_buy + ' {} {}'.format(pt[0], pt[1])
            for pt in self.keyPoint_cashier:
                str_cas = str_cas + ' {} {}'.format(pt[0], pt[1])
            str_buy +='\n'
            fp.write(str_buy)
            fp.write(str_cas)

    # load roi information from txt file
    def load_roi(self):
        # check existance.
        if not os.path.exists(self.roi_fn):
            return False, None

        #open
        with open(self.roi_fn, 'r') as fp:
            lines = fp.readlines()

        # check line number
        if len(lines) < 2:
            return False, None

        # convert raw data to numpy array
        strip1 = lines[0].strip().split()
        strip2 = lines[1].strip().split()

        strip1 = np.array([[int(x) for x in strip1]])
        strip2 = np.array([[int(x) for x in strip2]])

        nds_buyer = np.reshape(strip1, (1, -1, 2))
        nds_cashier= np.reshape(strip2, (1, -1, 2))

        # make roi mask
        h, w, _ = self.img.shape
        mask_buyer = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask_buyer, nds_buyer, 255)

        #cashier mask
        mask_cashier = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask_cashier, nds_cashier, 255)


        cv2.imwrite('buyer.jpg', mask_buyer)
        cv2.imwrite('cashier.jpg', mask_cashier)
        return True, (nds_buyer, mask_buyer, nds_cashier, mask_cashier)

    # mouse callback function
    def mouse_cb(self, event, x, y, flag):

        if event == cv2.EVENT_LBUTTONDOWN :
            self.lock.acquire()
            try:
                self.keyPoint.append((x, y))
            finally:
                self.lock.release()

            self.draw_region()
            return

    # draw roi region on initial image
    def draw_region(self):

        # draw current roi region
        key_len = len(self.keyPoint)
        if key_len <= 0:
            return
        clrLine = self.CLR_LINE
        clrCirle = self.CLR_LINE_S

        if self.bState:
            clrLine = self.CLR_LINE_S
            clrCirle = self.CLR_LINE

        temp = self.img.copy()
        if key_len >= 2:
            for cnt in range(1, key_len):
                cv2.line(temp, self.keyPoint[cnt - 1], self.keyPoint[cnt], clrLine, self.thick)
            cv2.line(temp, self.keyPoint[key_len - 1], self.keyPoint[0], clrLine, self.thick)

        for pt in self.keyPoint:
            cv2.circle(temp, pt, self.radius, clrCirle, -1)

        if self.bState: # completed buyer's region, so draw the buyer's regoin
            clrLine = self.CLR_LINE
            clrCirle = self.CLR_LINE_S

            key_len = len(self.keyPoint_buyer)
            if key_len >= 2:
                for cnt in range(1, key_len):
                    cv2.line(temp, self.keyPoint_buyer[cnt - 1], self.keyPoint_buyer[cnt], clrLine, self.thick)
                cv2.line(temp, self.keyPoint_buyer[key_len - 1], self.keyPoint_buyer[0], clrLine, self.thick)

            for pt in self.keyPoint_buyer:
                cv2.circle(temp, pt, self.radius, clrCirle, -1)

        cv2.imshow(self.winname, temp)


