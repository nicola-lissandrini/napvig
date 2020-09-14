#!/usr/bin/env python

import signal
import rospy
import numpy as np

from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Pose2D, Vector3
from napvig.msg import SearchHistory
import matplotlib.pyplot as plt

meas_topic = "/measures"
map_topic = "/map_values"
setpt_topic = "/setpoint"
history_topic = "/search_history"
grad = False

def handle_close (fig):
    quit ()

def sub (a, b):
    ret = Vector3 ()
    ret.x = a.x - b.x
    ret.y = a.y - b.y
    ret.z = a.z - b.z
    return ret

class DisplayNode:
    def measures_callback (self, measures):
        self.meas_np = np.array (measures.data)
        self.meas_np = np.reshape (self.meas_np, (measures.layout.dim[0].size, measures.layout.dim[1].size))
        self.sync_meas = True
    

    def map_callback (self, map_values):
        self.ranges = map_values.data[0:3]
        
        if (not grad):
            self.values = np.transpose (np.array (map_values.data[3:]).reshape (map_values.layout.dim[0].size, map_values.layout.dim[1].size))
        else:
            self.grads = np.array (map_values.data[3:]).reshape (map_values.layout.dim[0].size * map_values.layout.dim[1].size, map_values.layout.dim[2].size)

        self.draw ()
        self.sync_map = True

    def setpt_callback (self, setpt):
        self.setpt = setpt

    def history_callback (self, history):
        self.history = history

    def __init__ (self):
        rospy.init_node ("display_measures")

        self.fig = plt.figure ()
        self.meas_sub = rospy.Subscriber (meas_topic, Float64MultiArray, self.measures_callback, queue_size=1)
        self.map_sub = rospy.Subscriber (map_topic, Float64MultiArray, self.map_callback, queue_size=1)
        self.setpt_sub = rospy.Subscriber (setpt_topic, Pose2D, self.setpt_callback, queue_size=1)
        self.history_sub = rospy.Subscriber (history_topic, SearchHistory, self.history_callback, queue_size=1)
        self.sync_meas = False
        self.sync_map = False
        
        self.fig.canvas.mpl_connect('close_event', handle_close)


        self.meas_np = None
        self.values = None 
        self.grads = None
        self.setpt = None

        plt.show ()

    def draw_turtle (self, pos, direction, color="r", arrow_color="k"):
        plt.scatter (pos.x, pos.y, 4.5, c=color)
        plt.quiver (pos.x, pos.y, direction.x, direction.y, color=arrow_color)

    def draw_history (self):
        #self.history = SearchHistory ()
        j = 0
        
        for curr_path in self.history.triedPaths:
            last_pos = Vector3 ()
            self.draw_turtle (last_pos, self.history.initialSearch[j])
            for curr_pos in curr_path.poses:
                self.draw_turtle (curr_pos.pose.position, sub (curr_pos.pose.position, last_pos))
                last_pos = curr_pos.pose.position

    def draw(self):
        if (self.meas_np is None or (self.values is None and self.grads is None) ):
            return

        if (not self.sync_meas or not self.sync_map):
            return
        
        self.sync_map = False
        self.sync_meas = False
        
        xy_range = np.arange (self.ranges[0], self.ranges[1], self.ranges[2])
        x, y = np.meshgrid (xy_range, xy_range)

        self.fig.clear ()
        if (not grad):
            plt.pcolormesh (x, y, self.values, edgecolors="none", antialiased=True )
        else:
            plt.quiver(y, x, self.grads[:,0], self.grads[:,1], minshaft=0.1)

        setpt_dir = Pose2D ()
        pos_zero = Pose2D ()
        setpt_zero = Pose2D ()

        setpt_dir.x = np.cos (self.setpt.theta)
        setpt_dir.y = np.sin (self.setpt.theta)

        setpt_zero.x = 1
        setpt_zero.y = 0

        self.draw_turtle (self.setpt, setpt_dir)
        self.draw_turtle (pos_zero, setpt_zero, arrow_color="g")

        self.draw_history ()

        plt.scatter (self.meas_np[:,0], self.meas_np[:,1],2.5)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.draw ()
 
    def spin(self):
        rospy.spin ()

def signal_handler (sig, frame):
    plt.close ()


if __name__ == "__main__":
    try:
        signal.signal(signal.SIGINT, signal_handler)
        dn = DisplayNode()
        dn.spin ()

    except rospy.ROSInterruptException:
        pass