import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from SC_navigation.point_cloud2 import read_points
import laser_geometry.laser_geometry as lg


class LaserScanner():
    def __init__(self):
        self.lp = lg.LaserProjection()
        self.OFFSET = 20

    def trim(self,scanlist):
        trimmed_scanlist = np.delete(scanlist,range(self.OFFSET),0)
        trimmed_scanlist = np.delete(trimmed_scanlist, range( len(trimmed_scanlist) -self.OFFSET , len(trimmed_scanlist)),0)
        return trimmed_scanlist
    def get_obs_points(self,scan_msg=None):
        if scan_msg==None:
            scan_msg=rospy.wait_for_message("scan_raw",LaserScan, timeout=None)
        scan = self.lp.projectLaser(scan_msg)
        points=[]
        for p in read_points(scan, skip_nans=True):
            point=[p[0], p[1]]#, data[2], data[3]]
            points.append(point)
        trimmed_list=self.trim(points)
        return trimmed_list

    