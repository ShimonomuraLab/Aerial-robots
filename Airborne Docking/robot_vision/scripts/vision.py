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

import message_filters
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from robot_vision.msg import Img_results

class image_processing:

    def __init__(self):
        self.threshold = 37

        self._fps = rospy.get_param('/vision/_fps')

        self._imshow = rospy.get_param('/vision/_imshow')
        self._show_mask = rospy.get_param('/vision/_show_mask')

        self._set_threshold = rospy.get_param('/vision/_set_threshold')

        self._use_rcsize = rospy.get_param('/vision/_use_rcsize')
        self._plot = rospy.get_param('/vision/_plot')

        self._show_histgram = rospy.get_param('/vision/_show_histgram')

        self._VideoWriter = rospy.get_param('/vision/_VideoWriter')


        self.results = Img_results()

        self.createTrackbar = True

        self.color = np.zeros((480, 640, 3), dtype=np.uint8)
        self.depth = np.zeros((480, 640), dtype=np.uint16)

        self.seq = [0, 0]


        self.Countdown, self.results.reset = False, False

        self.start_Countdown = 0.

        self.width_pts, self.distance_pts, self.count_pts, self.poly, self.Object_size = np.zeros(100,), np.zeros(100,), np.zeros(16,), np.zeros(1,), 0
        self.at_close_range, self.found_Depth_Solution = False, False

        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out_color = cv2.VideoWriter('color.avi',self.fourcc, 25.0, (640,480))
        self.out_depth = cv2.VideoWriter('depth.avi',self.fourcc, 25.0, (640,480))


        self.bridge = CvBridge()
        self.depth_sub = rospy.Subscriber("/camera/depth_registered/sw_registered/image_rect_raw", Image, self.depth_callback)
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.color_callback)

        self.reset_sub = rospy.Subscriber("reset", Img_results, self.reset_callback)

        self.results_pub = rospy.Publisher("image_results", Img_results, queue_size=10)



    def reset_callback(self, msg):
        if msg.reset: self.all_reset('global')

    def color_callback(self, color_image):
        try:
            self.color = self.bridge.imgmsg_to_cv2(color_image, "bgr8")

        except CvBridgeError as e:
            print(e)

        self.seq[0] = color_image.header.seq

    def depth_callback(self, depth_image):
        try:
            self.depth = self.bridge.imgmsg_to_cv2(depth_image, "16UC1")

        except CvBridgeError as e:
            print(e)

        self.seq[1] = depth_image.header.seq




    def nothing(self, x): pass

    def all_reset(self, str):

        self.Countdown = False

        self.at_close_range, self.found_Depth_Solution = False, False
        self.width_pts, self.distance_pts, self.count_pts, self.poly, self.Object_size = np.zeros(100, ), np.zeros(100, ), np.zeros(16, ), np.zeros(self.poly), 0

        self.results.x, self.results.y, self.results.width, self.results.height, self.results.angle, self.results.distance = 320, 240, 0, 0, 0, 0

        print colored('Resetting.....', 'cyan', attrs=['bold'])

        if str is 'local':
            self.results.reset = True
            self.results_pub.publish(self.results)

        self.results.reset = False


    def set_threshold(self):
        if self.createTrackbar:
            cv2.namedWindow('Binary', cv2.WINDOW_AUTOSIZE)
            cv2.createTrackbar('threshold', 'Binary', self.threshold, 255, self.nothing)
            self.createTrackbar = False

        self.threshold = cv2.getTrackbarPos('threshold', 'Binary')



    def img_process(self):
        if self.color.shape[0] > 0 and self.depth.shape[0] > 0:
            color = self.color
            depth = self.depth

            Detected, self.results.tracking = False, False

            depth_mask = np.zeros((480, 640), dtype=np.uint8)
            mask = np.zeros((480, 640), dtype=np.uint8)
            if self._show_histgram: img_histgram = np.zeros([300, 256]).astype("uint8")

            depth_img = cv2.convertScaleAbs(depth, alpha=0.05)  # 255/5100

            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

            if self._set_threshold:
                if self._imshow: self.set_threshold()
                dst = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY_INV)[1]

            else:
                dst = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


            if self.at_close_range: Detected = True
            else:
                hist = cv2.calcHist([depth_img], [0], dst, [250], [4, 254])

                if np.max(hist) >= 100:

                    for i in xrange(250):
                        if hist[i] >= 100:
                            nearest = i + 1

                            Detected = True

                            if self._show_histgram:
                                cv2.line(img_histgram, (i, img_histgram.shape[0]),
                                         (i, img_histgram.shape[0] - img_histgram.shape[0] * (hist[i] / np.max(hist))),
                                         (255, 255, 255))
                            break


            if Detected:
                if self.at_close_range:
                    depth_mask = cv2.inRange(depth_img, 0, 30)    #0-0.7m
                    depth_mask = cv2.erode(depth_mask, None, iterations=6)
                    depth_mask = cv2.dilate(depth_mask, None, iterations=4)
                else:
                    depth_mask = cv2.inRange(depth_img, nearest, nearest + 20)
                    depth_mask = cv2.dilate(depth_mask, None, iterations=4)
                    depth_mask = cv2.erode(depth_mask, None, iterations=2)

                mask = cv2.bitwise_and(dst, depth_mask)

                cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

                if len(cnts) > 0 and not self._show_mask:
                    c = max(cnts, key=cv2.contourArea)
                    rect = cv2.minAreaRect(c)

                    if all(rect[1][i] >= 3 for i in range(2)): self.results.tracking = True

                    box = np.int0(cv2.boxPoints(rect))
                    cv2.drawContours(color, [box], -1, (0, 255, 0), 3)



            if self.results.tracking and not self._show_mask:

                self.results.x, self.results.y, self.results.width, self.results.height, self.results.angle = rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]

                if any(box[x][0] > 560 for x in range(4)) and any(box[x][0] < 20 for x in range(4)): self.results.x = 320
                if any(box[x][1] > 465 for x in range(4)) and any(box[x][1] < 20 for x in range(4)): self.results.y = 240

                distance_roi = depth[int(rect[0][1] - 2): int(rect[0][1] + 3),
                               int(rect[0][0] - 2): int(rect[0][0] + 3)]
                distance = np.median(distance_roi[np.nonzero(distance_roi)]) * 0.001  # find a reliable distance data

                if distance <= 0.65:
                    self.at_close_range = True  # Realsense r200 only can scan subjects between a distance of 0.5m - 3m(Note: Maximum distance is very dependent upon the module and the lighting).
                # When the object gets close, we want to know.
                elif distance > 0.65 and self.at_close_range:
                    self.at_close_range = False

                if not math.isnan(distance):
                    self.results.distance = distance
                    self.Countdown = False

                elif self.at_close_range and self._use_rcsize and self.found_Depth_Solution:  # When the object gets too close, use color image to calculate distance
                    self.results.distance = self.Object_size / rect[1][0]

                    if self.results.distance > 1 or self.results.distance < 0: self.results.distance = 0
                    print self.results.distance


                elif not self._use_rcsize or not self.found_Depth_Solution:

                    if self.Countdown:  # can't get distance, wait for 10sec and restart
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

                if distance >= 0.5 and distance <= 2 and self._use_rcsize and not self.found_Depth_Solution:  # Depth Solution
                    add_data = False
                    n = 1

                    for i in range(5, 20):
                        d = i * 0.1  # 0.5m to 2m

                        if distance > d and distance <= d + 0.1 and self.count_pts[n] < 5:
                            add_data = True
                            self.count_pts[n] += 1

                            print colored('Distance: %g to %g Recorded in %d' % (d, d + 0.1, self.count_pts[0]),
                                          'white', attrs=['bold'])
                            break
                        n += 1

                    if add_data:
                        self.width_pts[self.count_pts[0]] = rect[1][0]
                        self.distance_pts[self.count_pts[0]] = distance
                        self.count_pts[0] += 1

                        if self.count_pts[0] >= 30:  # enough data
                            _w = self.width_pts[np.nonzero(self.width_pts)]
                            d = self.distance_pts[np.nonzero(self.distance_pts)]
                            corr = abs(np.corrcoef(_w, d)[1, 0])  # correlation coefficients

                            if corr > 0.6:
                                self.Object_size = np.mean(_w * d)  # 633: Focal length in pixels
                                print colored('Found object size: %g m' % (self.Object_size / 633), 'magenta',
                                              attrs=['underline', 'bold'])

                                self.found_Depth_Solution = True

                                if self._plot:
                                    width_pl = np.linspace(10, 400, 100)

                                    plt.plot(_w, d, '.', width_pl, self.Object_size / width_pl, '-')

                                    plt.xlabel('Object size [pixels]')
                                    plt.ylabel('Distance [meter]')
                                    plt.xlim(0, 450)
                                    plt.ylim(0, 3)
                                    # plt.show()
                                    plt.savefig('lr.png')

                            else:
                                self.width_pts, self.distance_pts, self.count_pts, self.poly = np.zeros(
                                    100, ), np.zeros(100, ), np.zeros(16, ), np.zeros(self.poly)
                                print colored('The absolute value of the correlation coefficient is less than 0.6.',
                                              'yellow')
                                print colored('Redo.', 'yellow')

            else:
                self.results.x, self.results.y, self.results.width, self.results.height, self.results.angle, self.results.distance = 320, 240, 0, 0, 0, 0

            self.results_pub.publish(self.results)

            if self._imshow:
                if self._set_threshold: cv2.imshow("Binary", dst)
                if self._show_histgram: cv2.imshow("Mask", img_histgram)

                if self._show_mask:
                    cv2.imshow("Mask", depth_mask)

                else:
                    cv2.imshow("Color", color)
                    dep = cv2.convertScaleAbs(depth, alpha=0.12)
                    dep = cv2.applyColorMap(dep, cv2.COLORMAP_JET)
                    cv2.imshow("depth_img", dep)
                    cv2.imshow("depth_mask", depth_mask)
                    cv2.imshow("Mask", mask)


                k = cv2.waitKey(1)
                if k == ord('r'): self.all_reset('local')
                if k == ord('s'):
                    cv2.imwrite('depth_mask.bmp', depth_mask)
                    cv2.imwrite('Binary.bmp', dst)
                    cv2.imwrite('Color.bmp', color)
                    cv2.imwrite('depth_img.bmp', dep)
                    cv2.imwrite('mask.bmp', mask)
                    print colored('saved', 'yellow')

            if self._VideoWriter:
                self.out_color.write(color)
                #dep = cv2.cvtColor(depth_img, cv2.COLOR_GRAY2BGR)
                self.out_depth.write(dep)






def main(args):
    rospy.init_node('image_processing', anonymous=True)

    ip = image_processing()

    r = rospy.Rate(1000)
    seq = [0,0]

    while not rospy.is_shutdown():
        seq[0] = ip.seq[0]
        seq[1] = ip.seq[1]

        start = rospy.get_rostime()
        ip.img_process()

        if ip._fps:
            end = rospy.get_rostime()
            fps = 1./((end - start).to_sec())
            print fps

        while  not rospy.is_shutdown():
            if seq[0] != ip.seq[0] and seq[1] != ip.seq[1]:
                break

        r.sleep()


    cv2.destroyAllWindows()
    ip.out_color.release()
    ip.out_depth.release()
    #plt.close()

if __name__ == '__main__':
    main(sys.argv)

