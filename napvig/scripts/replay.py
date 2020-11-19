#!/usr/bin/env python3


import rospy 
import rospkg

from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState

def main():
    rospy.init_node('set_pose')
    sub = rospy.Subscriber ("/odom", Odometry, odomCallback)
    rospy.spin ()

def odomCallback (odomMsg):
    state_msg = ModelState()
    state_msg.model_name = 'turtlebot3_waffle'
    state_msg.pose = odomMsg.pose.pose
    #state_msg.twist = odomMsg.twist.twist
    rospy.wait_for_service('/gazebo/set_model_state')
    try:
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        resp = set_state( state_msg )

    except (rospy.ServiceException, e):
        print ("Service call failed: %s" % e)

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass