#!/usr/bin/env python

import roslib
roslib.load_manifest('robot_vision')
import sys
import rospy
import cv2
import numpy as np
import math

from termcolor import colored
import matplotlib.pyplot as plt

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from robot_vision.msg import Img_results

class image_processing:

    def __init__(self):
        self.bridge = CvBridge()
        self.depth_sub = rospy.Subscriber("/camera/depth_registered/sw_registered/image_rect_raw",Image,self.depth_callback)
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw",Image,self.color_callback)
        self.reset_sub = rospy.Subscriber("reset", Img_results, self.reset_callback)

        self.results_pub = rospy.Publisher("image_results",Img_results, queue_size = 10)

        self._fps = rospy.get_param('/vision/_fps')

        self._imshow = rospy.get_param('/vision/_imshow')
        self._show_depth_mask = rospy.get_param('/vision/_show_depth_mask')

        self._unknown_color = rospy.get_param('/vision/_unknown_color')
        self.color_set = rospy.get_param('/vision/set_color')

        self._use_rcsize = rospy.get_param('/vision/_use_rcsize')
        self._try_linear_regression = rospy.get_param('/vision/_try_linear_regression')
        self._plot = rospy.get_param('/vision/_plot')


        self.results = Img_results()

        self.color = np.zeros((480,640,3), dtype=np.uint8)
        self.depth = np.zeros((480,640), dtype=np.uint16)

        self.LowDepthThreshold,self.HighDepthThreshold, self.DetectionThreshold, self.minSize = 510, 3000, 100, 100

        self.roi_size = 10
        self.count, self.center = [0,5], [0,0]
        self.results.tracking, self.Countdown, self.results.reset = False, False, False
        self.roi_hist,self.track_window = [],[]
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1) # Setup the termination criteria, either 10 iteration or move by atleast 1 pt

        self.start_Countdown = 0.

        self.width_pts, self.distance_pts, self.count_pts, self.poly, self.Object_size = np.zeros(100,), np.zeros(100,), np.zeros(16,), np.zeros(1,), 0
        self.at_close_range, self.found_Depth_Solution = False, False



    def reset_callback(self, msg):
        if msg.reset: self.all_reset('global')

    def color_callback(self,color_image):
        try:
            self.color = self.bridge.imgmsg_to_cv2(color_image, "bgr8")
        except CvBridgeError as e:
            print(e)


    def depth_callback(self,depth_image):
        try:
            self.depth = self.bridge.imgmsg_to_cv2(depth_image, "16UC1")
        except CvBridgeError as e:
            print(e)

        (rows,cols) = self.depth.shape
        if cols > 60 and rows > 60 :

            self.img_process()



    def all_reset(self, str):

        self.count, self.center = [0, 6], [0, 0]
        self.results.tracking, self.Countdown = False, False

        self.at_close_range, self.found_Depth_Solution = False, False
        self.width_pts, self.distance_pts, self.count_pts, self.poly, self.Object_size = np.zeros(100, ), np.zeros(100, ), np.zeros(16, ), np.zeros(self.poly), 0

        self.results.x, self.results.y, self.results.width, self.results.height, self.results.angle = 0, 0, 0, 0, 0

        print colored('Resetting.....', 'cyan', attrs=['bold'])

        if str is 'local':
            self.results.reset = True
            self.results_pub.publish(self.results)

        self.results.reset = False



    def img_process(self):

        if self._fps: start = rospy.get_rostime()


        if self.count[1] > 0 and not self.results.tracking or self._show_depth_mask:         #object detection

            if self._unknown_color or self._show_depth_mask:                                       #detect the nearest object
                depth_mask = np.zeros(self.depth.shape, dtype=np.uint8)  #Create a black background

                depth_1d = self.depth.flat
                depth_hist, depth_bins = np.histogram(depth_1d, bins = np.arange(self.LowDepthThreshold, self.HighDepthThreshold + 1, 10))

                if np.max(depth_hist) >= self.minSize:
                    idx = np.where(depth_hist >= self.minSize)[0][0]
                    nearest = depth_bins[idx]

                    index_bool = np.logical_and(self.depth >= nearest, self.depth <= nearest + self.DetectionThreshold)    # boolean array
                    depth_mask[index_bool] = 255    # binarization

                kernel = np.ones((8, 8), np.uint8)
                #depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_OPEN, kernel)
                depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_CLOSE, kernel)    # closing small black points on the object

                cnts = cv2.findContours(depth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]    # find contours


            else:
                hsv = cv2.cvtColor(self.color, cv2.COLOR_BGR2HSV)        #Color Detection

                if self.color_set == 'red': mask = cv2.inRange(hsv, (0, 43, 46), (10, 255, 255))
                elif self.color_set == 'blue': mask = cv2.inRange(hsv, (100, 43, 46), (124, 255, 255))

                mask = cv2.erode(mask, None, iterations=2)
                mask = cv2.dilate(mask, None, iterations=2)

                cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]


            if len(cnts) > 0:
                c = max(cnts, key=cv2.contourArea)
                M = cv2.moments(c)

                if M['m00'] in range(400, 153600):    # if the area is 20*20 to 640*480/2
                    self.center = [int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])]

                    if not self._show_depth_mask:
                        self.count[1] -= 1

                        if self._unknown_color: print colored('Depth Detection Countdown: %d' %self.count[1], 'white', attrs=['bold'])
                        else: print colored('Color Detection Countdown: %d' %self.count[1], 'white', attrs=['bold'])

                        cv2.circle(self.color, (self.center[0], self.center[1]), 20, (0, 0, 255), -5)



        if self.count[1] == 0 and not self.results.tracking:

            if self.center[0] in range(40,600) and self.center[1] in range(40,440):

                self.track_window = (self.center[0] - self.roi_size, self.center[1] - self.roi_size, self.roi_size*2, self.roi_size*2)    # setup initial location of window (c,r,w,h)
                roi = self.color[self.center[1] - self.roi_size : self.center[1] + self.roi_size, self.center[0] - self.roi_size : self.center[0] + self.roi_size]    # set up the ROI for tracking [r:r+h, c:c+w]

                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv_roi, np.array((0., 43., 46.)), np.array((180., 255., 255.)))

                self.roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
                cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)

                self.results.tracking = True

                print colored('Start Tracking Object.', 'green', attrs=['bold'])

            else: self.count[1] = 6


        if self.results.tracking:

            hsv = cv2.cvtColor(self.color, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv],[0],self.roi_hist,[0,180],1)

            ret, self.track_window = cv2.CamShift(dst, self.track_window, self.term_crit)    # apply meanshift to get the new location

            if all(ret[1][i] <= 3 for i in range(2)):
                print colored('Missing target. Restarting.', 'red', attrs=['bold'])
                self.all_reset('local')
                return

            else: self.results.x, self.results.y, self.results.width, self.results.height, self.results.angle = ret[0][0], ret[0][1], ret[1][0], ret[1][1], ret[2]


            cv2.ellipse(self.color, ret, (0, 0, 0), 3)    # Draw it on image. ret = ((center_x,center_y),(width,height),angle)    ret[0][0],ret[0][1]
            #cv2.ellipse(self.depth, ret, (0, 0, 0), 3)


            distance_roi = self.depth[int(ret[0][1] - 2) : int(ret[0][1] + 3), int(ret[0][0] - 2) : int(ret[0][0] + 3)]
            distance = np.median(distance_roi[np.nonzero(distance_roi)])*0.001      #find a reliable distance data

            if distance <= 0.6: self.at_close_range = True   #Realsense r200 only can scan subjects between a distance of 0.5m - 3m(Note: Maximum distance is very dependent upon the module and the lighting).
                                                             #When the object gets close, we want to know.
            elif distance > 0.8 and self.at_close_range: self.at_close_range = False


            if not math.isnan(distance):
                self.results.distance = distance
                self.Countdown = False

            elif self.at_close_range and self._use_rcsize and self.found_Depth_Solution: #When the object gets too close, use color image to calculate distance
                if self._try_linear_regression: self.results.distance = self.poly(ret[1][0])
                else: self.results.distance = self.Object_size / ret[1][0]

                if self.results.distance > 1 or self.results.distance < 0: self.results.distance = 0
                print self.results.distance


            elif not self._use_rcsize or not self.found_Depth_Solution:

                if self.Countdown:        #can't get distance, wait for 10sec and restart
                    now_Countdown = rospy.get_rostime()
                    time = (now_Countdown - self.start_Countdown).to_sec()

                    if time > 1 and not self._use_rcsize: self.results.distance = 0

                    if time > 10:
                        print colored('Tracking failed. Restarting.', 'red', attrs=['reverse'])
                        self.all_reset('local')
                        return


                else:
                    self.start_Countdown = rospy.get_rostime()
                    self.Countdown = True



            if distance >= 0.5 and distance <= 2 and self.count[0] >= 5 and self._use_rcsize and not self.found_Depth_Solution:   #Depth Solution
                add_data = False
                n = 1

                for i in range(5, 20):
                    d = i * 0.1  # 0.5m to 2m

                    if distance > d and distance <= d + 0.1 and self.count_pts[n] < 5:
                        add_data = True
                        self.count_pts[n] += 1

                        print colored('Distance: %g to %g Recorded in %d' % (d, d + 0.1, self.count_pts[0]), 'white', attrs=['bold'])
                        break
                    n += 1

                if add_data:
                    self.width_pts[self.count_pts[0]] = ret[1][0]
                    self.distance_pts[self.count_pts[0]] = distance
                    self.count_pts[0] += 1

                    if self.count_pts[0] >= 40:    #enough data
                        _w = self.width_pts[np.nonzero(self.width_pts)]
                        d = self.distance_pts[np.nonzero(self.distance_pts)]
                        corr = abs(np.corrcoef(_w, d)[1,0])    #correlation coefficients

                        if corr > 0.6:
                            if self._try_linear_regression:
                                self.poly = np.poly1d(np.polyfit(_w , np.reciprocal(d), 1))       #Least squares polynomial fit
                                print colored('Successfully found linear regression equation.', 'magenta', attrs=['underline', 'bold'])

                            else:
                                self.Object_size = np.mean( _w * d )     # 633: Focal length in pixels
                                print colored('Found object size: %g m' % (self.Object_size / 633), 'magenta', attrs=['underline', 'bold'])


                            self.found_Depth_Solution = True

                            if self._plot:
                                width_pl = np.linspace(10, 400, 100)

                                if self._try_linear_regression: plt.plot(_w , d, '.', width_pl, np.reciprocal(self.poly(width_pl)), '-')     #plot the linear regression equation

                                else: plt.plot(_w , d, '.', width_pl, self.Object_size / width_pl, '-')

                                plt.xlabel('Object size [pixels]')
                                plt.ylabel('Distance [meter]')
                                plt.xlim(0, 450)
                                plt.ylim(0, 3)
                                #plt.show()
                                plt.savefig('lr.png')

                        else:
                            self.width_pts, self.distance_pts, self.count_pts, self.poly = np.zeros(100,), np.zeros(100,), np.zeros(16,), np.zeros(self.poly)
                            print colored('The absolute value of the correlation coefficient is less than 0.6.', 'yellow')
                            print colored('Redo.', 'yellow')

            self.count[0] += 1

        self.results_pub.publish(self.results)




        if self._imshow:
            #depth_img = cv2.convertScaleAbs(self.depth,0.063)
            #cv2.imshow("Depth", depth_img)

            if self._show_depth_mask: cv2.imshow("Depth_Detection", depth_mask)

            else: cv2.imshow("Color", self.color)

            k = cv2.waitKey(3)
            if k == ord('r'): self.all_reset('local')
            if k == ord('c'): self.count[1], self.results.tracking, self.center = 0, False, [320,240]

        if self._fps:
            end = rospy.get_rostime()
            fps = 1./((end - start).to_sec())
            print fps

def main(args):
    ip = image_processing()

    rospy.init_node('image_processing', anonymous=True)

    try:
        rospy.spin()

    except KeyboardInterrupt:
        print("Shutting down vision module")

    cv2.destroyAllWindows()
    #plt.close()

if __name__ == '__main__':
    main(sys.argv)

