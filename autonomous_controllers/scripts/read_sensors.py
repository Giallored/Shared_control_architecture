#!/usr/bin/env python
import rospy
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from src.utils import FakeLaserScanner,Obstacles,TIAgo
from gazebo_msgs.msg import ModelStates
import numpy as np
from scipy.spatial.transform import Rotation


class Sensor():
    def __init__(self, poly_degree=3, n_actions=20,rate=10):
        #self.laser_scanner=Fake_LaserScanner()
        self.tiago=TIAgo()
        self.scanner=FakeLaserScanner()
        #self.subscriber = rospy.Subscriber("/gazebo/model_states",ModelStates, self.callback)

    def main(self):
        rospy.Subscriber("/gazebo/model_states",ModelStates, self.callback)
        rospy.spin() # spin() simply keeps python from exiting until this node is stopped

    def callback(self,data):
        tiago_pos = data.pose[-1].position
        tiago_or = data.pose[-1].orientation
        
        self.tiago.set_MBpose(tiago_pos,tiago_or)
        obs_ids= data.name[:-1]
        obs_poses = data.pose[:-1]

        obstacles = Obstacles(obs_ids,obs_poses)
        visible_obs_pos,visible_obs_id = self.scanner.get_visible_obs(self.tiago,obstacles)
        print(len(visible_obs_id),'obstacles are visible')
        print(visible_obs_id)
        print('------------------')


if __name__ == '__main__':
    try:
        rospy.init_node('sensor_node', anonymous=True)
        sensor = Sensor()
        sensor.main()
    except rospy.ROSInterruptException:
        pass