#!/usr/bin/env python

import sensor_msgs.point_cloud2 as pc2
import rospy
from sensor_msgs.msg import PointCloud2, LaserScan
import laser_geometry.laser_geometry as lg
import math

class Sensor():
    def __init__(self, poly_degree=3, n_actions=20,rate=10):
        #self.laser_scanner=Fake_LaserScanner()
        self.pub = rospy.Publisher("autonomous_controllers/obs_pos", PointCloud2, queue_size=1)
        self.lp = lg.LaserProjection()

        self.rate=rospy.Rate(rate) # 10hz



    def main(self):
        rospy.Subscriber("scan_raw",LaserScan, self.callback)
        
        rospy.spin() # spin() simply keeps python from exiting until this node is stopped

    def callback(self,data):
        # convert the message of type LaserScan to a PointCloud2
        msg = self.lp.projectLaser(data)
        # now we can do something with the PointCloud2 for example:
        # publish it
        self.pub.publish(msg)
        #rospy.loginfo(msg)
        self.rate.sleep()
        # convert it to a generator of the individual points
        #point_generator = pc2.read_points(pc2_msg)

if __name__ == '__main__':
    try:
        rospy.init_node("laserscan_to_pointcloud", anonymous=True)
        sensor = Sensor()
        sensor.main()
    except rospy.ROSInterruptException:
        pass
