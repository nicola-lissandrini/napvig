#!/usr/bin/env python

import rospy
import numpy as np

from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Pose2D
import matplotlib.pyplot as plt

meas_topic = "/measures"
map_topic = "/map_values"
setpt_topic = "/setpoint"
grad = False

class DisplayNode:
    def measures_callback (self, measures):
        self.meas_np = np.array (measures.data)
        self.meas_np = np.reshape (self.meas_np, (measures.layout.dim[0].size, measures.layout.dim[1].size))

    def map_callback (self, map_values):
        self.ranges = map_values.data[0:3]
        
        if (not grad):
            self.values = np.transpose (np.array (map_values.data[3:]).reshape (map_values.layout.dim[0].size, map_values.layout.dim[1].size))
        else:
            self.grads = np.array (map_values.data[3:]).reshape (map_values.layout.dim[0].size * map_values.layout.dim[1].size, map_values.layout.dim[2].size)
        print ("DFgsdfg")
        self.draw ()

    def setpt_callback (self, setpt):
        self.setpt = setpt

    def __init__ (self):
        rospy.init_node ("display_measures")

        self.fig = plt.figure ()
        self.meas_sub = rospy.Subscriber (meas_topic, Float64MultiArray, self.measures_callback, queue_size=1)
        self.map_sub = rospy.Subscriber (map_topic, Float64MultiArray, self.map_callback, queue_size=1)
        self.setpt_sub = rospy.Subscriber (setpt_topic, Pose2D, self.setpt_callback, queue_size=1)

        self.meas_np = None
        self.values = None 
        self.grads = None
        self.setpt = None

        plt.show () 

    def draw(self):
        if (self.meas_np is None or (self.values is None and self.grads is None) ):
            return
        print ("auaa")
        xy_range = np.arange (self.ranges[0], self.ranges[1], self.ranges[2])
        x, y = np.meshgrid (xy_range, xy_range)

        self.fig.clear ()
        if (not grad):
            plt.pcolormesh (x, y, self.values, edgecolors="none", antialiased=True )
        else:
            plt.quiver(y, x, self.grads[:,0], self.grads[:,1], minshaft=0.1)
        plt.scatter (self.setpt.x, self.setpt.y, 4.5,c="r")
        plt.quiver (self.setpt.x, self.setpt.y, np.cos (self.setpt.theta), np.sin (self.setpt.theta))
        plt.scatter (self.meas_np[:,0], self.meas_np[:,1],2.5)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.draw ()
 
    def spin(self):
        rospy.spin ()

if __name__ == "__main__":
    try:
        dn = DisplayNode()
        dn.spin ()

    except rospy.ROSInterruptException:
        pass