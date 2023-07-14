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
        self.range_min= 0.05000000074505806
        self.range_max= 25.0

    def trim(self,list,offset):
        trimmed_list = np.delete(list,range(offset),0)
        trimmed_list = np.delete(trimmed_list, range( len(trimmed_list) -offset, len(trimmed_list)),0)
        return trimmed_list
        
    def get_obs_points(self,scan_msg=None):
        if scan_msg==None:
            scan_msg=rospy.wait_for_message("scan_raw",LaserScan, timeout=None)
        scan_msg = self.saturate(scan_msg)
        pointcloud = self.lp.projectLaser(scan_msg)
        ranges=self.trim(scan_msg.ranges,offset=77)
        points=[]
        for p in read_points(pointcloud, skip_nans=True):
            point=[p[0], p[1]]#, data[2], data[3]]
            points.append(point)
        points=self.trim(points,offset=20)
        #plt.clf()
        #plt.plot(trimmed_list[:,0],trimmed_list[:,1],'r.')
        #plt.show()
        return ranges.tolist(),points
    
    def saturate(self,scan_msg):
        n=len(scan_msg.ranges)
        new_ranges = []
        for i in range(n):
            r=scan_msg.ranges[i]
            if r>self.range_max:
                new_ranges.append(self.range_max)
            elif r<self.range_min:
                new_ranges.append(self.range_max)
            else:
                new_ranges.append(r)
        scan_msg.ranges=new_ranges
        return scan_msg

    