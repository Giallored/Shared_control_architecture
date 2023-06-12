#!/usr/bin/env python

import sensor_msgs.point_cloud2 as pc2
import rospy
from sensor_msgs.msg import PointCloud2, LaserScan
import laser_geometry.laser_geometry as lg
import math
from std_msgs.msg import Bool

class Sensor():
    def __init__(self, poly_degree=3, n_actions=20,rate=50):
        
        #self.laser_scanner=Fake_LaserScanner()
        self.pub = rospy.Publisher("autonomous_controllers/obs_pos", PointCloud2, queue_size=1)
        self.lp = lg.LaserProjection()
        self.scan_request=False
        self.rate=rospy.Rate(rate) # 10hz



    def main(self):
        while not rospy.is_shutdown():
            #print('-------------------------------------------------')
            rospy.Subscriber('autonomous_controllers/scan_request',Bool, self.callback)
            #scan_request=rospy.wait_for_message("autonomous_controllers/scan_request", Bool, timeout=True)
            #print('Scan_request = ',scan_request)
            #if scan_request:
            #    rospy.Subscriber("scan_raw",LaserScan, self.callback)
                #scan=rospy.wait_for_message("scan_raw",LaserScan, timeout=True)
                #self.callback(scan)  
            #rospy.Subscriber("scan_raw",LaserScan, self.drain)
            #rospy.spin() # spin() simply keeps python from exiting until this node is stopped

    def callback(self,data):
        print('Reiceived scan request!')
        scan_msg=rospy.wait_for_message("scan_raw",LaserScan, timeout=None)
        scan = self.lp.projectLaser(scan_msg)
        self.pub.publish(scan)
        print('Point cloud sent!')
        #print(scan)
        ##rospy.loginfo(msg)
        self.rate.sleep()
    
    def drain(self,data):
        print(' - Scan dismissed!')
        self.rate.sleep()
        #pass
        #print('NO scan request -> scan request drained')
        

if __name__ == '__main__':
    try:
        rospy.init_node("laserscan_to_pointcloud", anonymous=True)
        sensor = Sensor()
        sensor.main()
    except rospy.ROSInterruptException:
        pass
