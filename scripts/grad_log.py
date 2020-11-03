#!/usr/bin/env python3

import rospy
import numpy as np

from std_msgs.msg import Float64MultiArray
import matplotlib.pyplot as plt

grad_topic = "/grad_log"

dim = 1

class GradLogNode:
    def grad_callback (self, grad_log):
        self.grad_log = np.array (grad_log.data)
        if (dim == 2):
            self.grad_log = np.reshape (self.grad_log, (grad_log.layout.dim[0].size, grad_log.layout.dim[1].size))

        self.draw ()

    def __init__ (self):
        rospy.init_node ("grad_log")

        self.fig = plt.figure ()
        self.meas_sub = rospy.Subscriber (grad_topic, Float64MultiArray, self.grad_callback, queue_size=1)

        self.meas_np = None
        self.values = None 
        self.grads = None
        self.setpt = None

        plt.show () 

    def draw(self):
        if (self.grad_log is None):
            return
        self.fig.clear ()
        if (dim == 2):
            plt.plot (self.grad_log[:,0])
            plt.plot (self.grad_log[:,1])
        elif (dim == 1):
            plt.plot (self.grad_log)
        plt.grid ()
        plt.draw ()
 
    def spin(self):
        rospy.spin ()

if __name__ == "__main__":
    try:
        dn = GradLogNode()
        dn.spin ()

    except rospy.ROSInterruptException:
        pass