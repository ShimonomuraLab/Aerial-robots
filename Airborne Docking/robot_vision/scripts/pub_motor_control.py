#!/usr/bin/python
import rospy
from robot_vision.msg import Motor

def pub_motor_control():
    rospy.init_node('pub_motor_control', anonymous=True)

    pub = rospy.Publisher('motor_control', Motor, queue_size=2)
    r = rospy.Rate(100)

    msg = Motor()

    msg.motor[0] = 400
    msg.motor[1] = 400
    msg.motor[2] = 1000
    msg.motor[3] = 1000

    while not rospy.is_shutdown():
        msg.motor[0] += 1
        msg.motor[1] += 2

        if msg.motor[0] > 700 or msg.motor[1] > 700:
            msg.motor[0] = 400
            msg.motor[1] = 400

        pub.publish(msg)
        r.sleep()

if __name__ == '__main__':
    try:
            pub_motor_control()
    except rospy.ROSInterruptException: pass
