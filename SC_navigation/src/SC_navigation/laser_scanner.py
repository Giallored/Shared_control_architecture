        

import laser_geometry.laser_geometry as lg
import rospy
from auto_controller.point_cloud2 import read_points
import numpy as np









class LaserScan:
    def __init__(
        self,
        time, frame_id,
        angle_min, angle_max, angle_increment,
        time_increment,
        scan_time,
        range_min, range_max,
        ranges, intensities):
        self.time            = time
        self.frame_id        = frame_id
        self.angle_min       = angle_min
        self.angle_max       = angle_max
        self.angle_increment = angle_increment
        self.range_min       = range_min
        self.range_max       = range_max
        self.ranges          = ranges
        self.intensities     = intensities

    @staticmethod
    def from_message(laser_scan_msg):
        return LaserScan(
            laser_scan_msg.header.stamp.to_sec(),
            laser_scan_msg.header.frame_id,
            laser_scan_msg.angle_min,
            laser_scan_msg.angle_max,
            laser_scan_msg.angle_increment,
            laser_scan_msg.time_increment,
            laser_scan_msg.scan_time,
            laser_scan_msg.range_min,
            laser_scan_msg.range_max,
            laser_scan_msg.ranges,
            laser_scan_msg.intensities
        )