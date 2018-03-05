#!/usr/bin/python

import rospy
import numpy as np
import math
from tf import transformations

from robot_vision.msg import Motor
from robot_vision.msg import Img_results
from robot_vision.msg import UAV_Position
from sensor_msgs.msg import Imu

class self_localization:

    def __init__(self):
        self.rs_pub = rospy.Publisher('reset', Img_results, queue_size=10)
        self.pos_pub = rospy.Publisher('position', UAV_Position, queue_size=10)
        self.img_sub = rospy.Subscriber("image_results", Img_results, self.img_callback)
        self.motor_sub = rospy.Subscriber('motor_position', Motor, self.motor_callback)
        self.imu_sub = rospy.Subscriber('dji_sdk/imu', Imu, self.imu_callback)

        self.min_Angle, self.max_Angle = rospy.get_param('/Pantilt_control/min_Angle'), rospy.get_param('/Pantilt_control/max_Angle')
        self.Default_Angle_Pan = rospy.get_param('/self_localization/Default_Angle_Pan')
        self.Default_Angle_Tilt = rospy.get_param('/self_localization/Default_Angle_Tilt')

        self.Position_xy = rospy.get_param('/self_localization/Position_xy')
        self.Position_Height_Min = rospy.get_param('/self_localization/Position_Height_Min')
        self.Position_Height_Max = rospy.get_param('/self_localization/Position_Height_Max')

        self.DegToRad, self.RadToDeg = 0.017453, 57.3248
        self.motor_Resolution = 0.29296

        self.rs = Img_results()
        self.rs.reset = False

        self.distance, self.angle, self.e = 0, [0,0], [0,0,0]


    def motor_callback(self,msg):
        if all(msg.motor[i] in range(self.min_Angle, self.max_Angle) for i in range(2)): self.angle = [msg.motor[1], msg.motor[0]]

        if any(msg.motor[i] == 0 for i in range(2)): pass

        elif not all(msg.motor[i] in range(self.min_Angle + 70, self.max_Angle - 50) for i in range(2)): self.all_reset()

    def img_callback(self, msg):
        self.distance = msg.distance
        if msg.tracking: self.localization()

    def imu_callback(self, msg):
        self.e = transformations.euler_from_quaternion((msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w))




    def all_reset(self):
        self.distance, self.angle, self.e = 0, [0,0], [0,0,0]

        self.rs.reset = True
        self.rs_pub.publish(self.rs)

        self.rs.reset = False

    def localization(self):
        pos = UAV_Position()

        Pan = self.angle[0] - self.Default_Angle_Pan
        Tilt = self.angle[1] - self.Default_Angle_Tilt

        pos.y = -math.sin( Tilt * self.motor_Resolution * self.DegToRad - self.e[0] ) * math.cos( Pan * self.motor_Resolution * self.DegToRad - self.e[1] ) * self.distance
        pos.x = math.sin( Pan * self.motor_Resolution * self.DegToRad - self.e[1] ) * math.cos( Tilt * self.motor_Resolution * self.DegToRad - self.e[0] ) * self.distance
        pos.z = math.cos( Pan * self.motor_Resolution * self.DegToRad - self.e[1] ) * math.cos( Tilt * self.motor_Resolution * self.DegToRad - self.e[0] ) * self.distance

        if pos.x < self.Position_xy and pos.y < self.Position_xy and pos.z <= self.Position_Height_Max and pos.z >= self.Position_Height_Min:
            self.pos_pub.publish(pos)
            print pos

        else:
            self.all_reset()
            return

        self.rs_pub.publish(self.rs)





def main():
    rospy.init_node('self_localization', anonymous=True)
    up = self_localization()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down UAV self localization module"

if __name__ == '__main__':
    main()
