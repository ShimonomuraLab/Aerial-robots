#!/usr/bin/python

import rospy
import numpy as np
import cv2
import sys

from robot_vision.msg import Motor
from robot_vision.msg import Img_results
from sensor_msgs.msg import Imu

class Pantilt_control:

    def __init__(self):
        self.av = [0., 0.]
        self.min_Angle, self.max_Angle = rospy.get_param('/Pantilt_control/min_Angle'), rospy.get_param('/Pantilt_control/max_Angle')
        self.min_Speed, self.max_Speed = rospy.get_param('/Pantilt_control/min_Speed'), rospy.get_param('/Pantilt_control/max_Speed')

        self.Kp = rospy.get_param('/Pantilt_control/Kp')
        self.Ki = rospy.get_param('/Pantilt_control/Ki')
        self.Kd = rospy.get_param('/Pantilt_control/Kd')
        self.Kdv = rospy.get_param('/Pantilt_control/Kdv')

        self._gain = rospy.get_param('/Pantilt_control/_gain')

        self.createTrackbar = True

        self.PxlToAngle = 0.37  #70/640(Pixel to Degree) /0.29(Motor Resolution)  Note: Realsense r200 Field of view (D * V * H):  77 * 43* 70

        self.Half_Frame_Width, self.Half_Frame_Height = 320, 240

        self.INT_TIME = 0.033
        self.motor_angle, self.pre_aErr, self.s_aErrIntg = [512, 512], [0., 0.], [0., 0.]

        self.motor_pub = rospy.Publisher('motor_control', Motor, queue_size=10)
        self.img_sub = rospy.Subscriber("image_results", Img_results, self.img_callback)
        self.reset_sub = rospy.Subscriber("reset", Img_results, self.reset_callback)
        self.imu_sub = rospy.Subscriber('dji_sdk/imu', Imu, self.imu_callback)



    def img_callback(self,msg):
        self.motor_control(msg)

    def reset_callback(self,msg):
        if msg.reset: self.all_reset()

    def imu_callback(self, msg):
        self.av[0] = msg.angular_velocity.y
        self.av[1] = msg.angular_velocity.x



    def nothing(self,x):
        pass

    def all_reset(self):
        mp = Motor()
        mp.read = False
        mp.motor = [512, 512, 500, 500]

        self.motor_pub.publish(mp)

        self.motor_angle, self.pre_aErr, self.s_aErrIntg = [512, 512], [0., 0.], [0., 0.]
        self.av = [0., 0.]


    def gain(self):
        if self.createTrackbar:
            cv2.namedWindow('Gain', cv2.WINDOW_NORMAL)
            cv2.createTrackbar('Kp', 'Gain', int(self.Kp * 1000), 300, self.nothing)
            cv2.createTrackbar('Ki', 'Gain', int(self.Ki * 1000), 250, self.nothing)
            cv2.createTrackbar('Kd', 'Gain', int(self.Kd * 1000), 250, self.nothing)
            cv2.createTrackbar('Kdv', 'Gain', int(self.Kdv * 10), 100, self.nothing)
            self.createTrackbar = False

        self.Kp = cv2.getTrackbarPos('Kp', 'Gain') / 1000.
        self.Ki = cv2.getTrackbarPos('Ki', 'Gain') / 1000.
        self.Kd = cv2.getTrackbarPos('Kd', 'Gain') / 1000.
        self.Kdv = cv2.getTrackbarPos('Kdv', 'Gain') / 10.

        if cv2.waitKey(1) & 0xFF == ord('s'):
            lines = open('/home/robot/catkin_ws/src/robot_vision/launch/PanTilt.launch', 'r').readlines()

            lines[11] = '    <param name="Kp" type="double" value="' + str(self.Kp) + '" />\n'
            lines[12] = '    <param name="Ki" type="double" value="' + str(self.Ki) + '" />\n'
            lines[13] = '    <param name="Kd" type="double" value="' + str(self.Kd) + '" />\n'
            lines[14] = '    <param name="Kdv" type="double" value="' + str(self.Kdv) + '" />\n'

            open('/home/robot/catkin_ws/src/robot_vision/launch/PanTilt.launch', 'w').writelines(lines)
            print('saved')



    def motor_control(self, img):
        if self._gain: self.gain()

        mp = Motor()
        mp.read = False

        if img.reset:
            self.all_reset()
            return

        elif img.tracking:
            aErr = [(img.x - self.Half_Frame_Width) * self.PxlToAngle,
                    (self.Half_Frame_Width - img.y) * self.PxlToAngle]

            s_aErrDerv, motor_speed = [0., 0.], [0, 0]

            for i in range(2):
                s_aErrDerv[i] =  ( aErr[i] - self.pre_aErr[i] ) / self.INT_TIME    # PID
                self.s_aErrIntg[i] += aErr[i] * self.INT_TIME

                self.motor_angle[i] = int( min ( max( self.motor_angle[i] + self.Kp * aErr[i] + self.Ki * self.s_aErrIntg[i] + self.Kd * s_aErrDerv[i] + self.Kdv * self.av[i] , self.min_Angle ), self.max_Angle) )

                motor_speed[i] = 1000
                self.pre_aErr[i] = aErr[i]  #abs( aErr[i] )

            mp.motor = [self.motor_angle[1], self.motor_angle[0], motor_speed[1], motor_speed[0]]   # [motor 1 position, motor 2 position, motor 1 speed, motor 2 speed]

        else:
            mp.motor = [self.motor_angle[1], self.motor_angle[0], 500, 500]

        self.motor_pub.publish(mp)


def main():
    rospy.init_node('Pantilt_control', anonymous=True)
    pt = Pantilt_control()

    try:
        rospy.spin()

    except KeyboardInterrupt:
        print "Shutting down Pantilt control"

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
