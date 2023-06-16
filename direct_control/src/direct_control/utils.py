import rospy
import time
import os

def get_output_folder(parent_dir, env_name):
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir

def countdown(n):
    for i in (range(n,0,-1)): 
        print(f"{i}", end="\r", flush=True)
        time.sleep(1)
    print('GO!')

def twist_to_cmd(msg):
    return [msg.linear.x,msg.angular.z]