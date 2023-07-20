import numpy as np
from scipy.spatial.transform import Rotation
from SC_navigation.utils import *


class TIAgo():
    def __init__(self, clear=0.2, mb_position=[0.,0.],mb_orientation=[0.,0.,0.]):
        self.mb_position=mb_position # wrt RFworld
        self.prev_mb_position=mb_position
        self.mb_orientation=mb_orientation # wrt RFworld
        self.mb_v = 0.0     # linear vel wrt RFworld
        self.mb_om = 0.0    # angular vel wrt RFworld
        self.clear = clear
        self.Tf_tiago_w = np.zeros((4,4))

    def set_MBpose(self,pose,vels):
        self.mb_prev_position=self.mb_position
        self.mb_position=pose.position
        self.mb_orientation=pose.orientation
        rot = Rotation.from_euler('xyz', self.mb_orientation, degrees=False)
        self.Tf_tiago_w = Pose2Homo(rot.as_matrix(),self.mb_position)
        self.mb_v = vels[0]
        self.mb_om = vels[1] 
        
    def get_relative_pos(self,obj_pos):
        pos = obj_pos+[1]
        rel_pos = np.dot(np.linalg.inv(self.Tf_tiago_w),pos)
        rel_pos=rel_pos[:-1]
        return rel_pos


class Object():
    def __init__(self, id:str, pos,theta, dist:float):
        self.id=id
        self.position=pos
        self.orientation=theta
        self.distance=dist