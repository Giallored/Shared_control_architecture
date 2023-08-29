import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from SC_navigation.point_cloud2 import read_points
import laser_geometry.laser_geometry as lg
import matplotlib.pyplot as plt



class LaserScanner():
    def __init__(self):
        self.lp = lg.LaserProjection()
        self.scan_msg = rospy.wait_for_message("scan_raw",LaserScan, timeout=None)
        self.OFFSET = 20
        self.figure=plt.figure()
        self.angle_min= -1.9198600053787231
        self.angle_max= 1.9198600053787231
        self.angle_increment= 0.005774015095084906
        self.range_min= 0.05000000074505806
        self.range_max= 25.0

    def trim(self,list,offset):
        trimmed_list = np.delete(list,range(offset),0)
        trimmed_list = np.delete(trimmed_list, range( len(trimmed_list) -offset, len(trimmed_list)),0)
        return trimmed_list
        
    def get_obs_points(self,max_dist = 3):
        msg = self.scan_msg
        #if not len(msg.ranges) == 666:
        #    msg = self.padding(msg)
        msg.ranges = self.trim(msg.ranges,offset=20)
        
        pointcloud = self.lp.projectLaser(msg)
        points=[]
        for p in read_points(pointcloud, skip_nans=True):
            point=[p[0], p[1]]#, data[2], data[3]]
            if np.linalg.norm(point)<=max_dist:
                points.append(point)

        #ranges = self.preproces(msg.ranges,max_dist)
        ranges,mask = self.get_mask(msg.ranges,max_dist)


        return ranges,mask,np.array(points)
    
    def padding(self,msg):
        n=len(msg.ranges)
        if n<666:
            padding = int((666-n)/2)
            cunck = [np.inf]*padding
            msg.ranges = [*cunck , *msg.ranges, *cunck]
        return msg
    

    def preproces(self,ranges,max_dist,max_span=np.pi):
        new_ranges = np.clip(ranges,0,max_dist)
        new_ranges = np.subtract(np.ones(new_ranges.shape)*max_dist,new_ranges)
        ranges2trim = max_span//self.angle_increment
        offset = int((new_ranges.shape[0] - ranges2trim)/2)
        new_ranges = self.trim(new_ranges,offset=offset)
        return new_ranges.tolist()
    
    def get_mask(self,ranges,max_dist):
        mask=[]
        for r in ranges:
            if r>max_dist:
                mask.append(0)
            else:
                mask.append(1)
        return ranges,mask

    
    

