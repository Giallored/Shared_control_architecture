import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from SC_navigation.point_cloud2 import read_points
import laser_geometry.laser_geometry as lg
import matplotlib.pyplot as plt



class LaserScanner():
    def __init__(self):
        self.lp = lg.LaserProjection()
        self.OFFSET = 20
        self.figure=plt.figure()

    def trim(self,list):
        trimmed_list = np.delete(list,range(self.OFFSET),0)
        trimmed_list = np.delete(trimmed_list, range( len(trimmed_list) -self.OFFSET , len(trimmed_list)),0)
        return trimmed_list
        
    def get_obs_points(self,scan_msg=None):
        if scan_msg==None:
            scan_msg=rospy.wait_for_message("scan_raw",LaserScan, timeout=None)
        scan = self.lp.projectLaser(scan_msg)
        ranges=self.trim(scan_msg.ranges)
        points=[]
        for p in read_points(scan, skip_nans=True):
            point=[p[0], p[1]]#, data[2], data[3]]
            points.append(point)
        points=self.trim(points)
        #plt.clf()
        #plt.plot(trimmed_list[:,0],trimmed_list[:,1],'r.')
        #plt.show()
        return ranges.tolist(),points

    