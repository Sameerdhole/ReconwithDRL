###Mapping#####
#1)Voxgraph csv writer
#or
#2)SLAM Realtime
import os, sys


path = os.path.expanduser('~\Documents\Airsim')
if not os.path.exists(path):
	os.makedirs(path)
md = path + '\voxgraph\map_data.csv'
map_data= open(md, 'w')

s_log = '{:<6s} - Level {:>2d} - Iter: {:>6d}/{:<5d} {:<8s}-{:>5s} lr: {:>1.8f} Ret = {:>+6.4f} Last Crash = {:<5d} t={:<1.3f} SF = {:<5.4f}  Reward: {:<+1.4f}  '.format(
                                        name_agent,
                                        int(level[name_agent]),
                                        iter[name_agent],
                                        epi_num[name_agent],
                                        action_word,
                                        action_type,
                                        algorithm_cfg.learning_rate,
                                        ret[name_agent],
                                        last_crash[name_agent],
                                        time_exec,
                                        distance[name_agent],
                                        reward)


map_data.write(s_log + '\n')


#1)Enable lidar in airsim
#2)https://github.com/ethz-asl/voxgraph make rosbag dataset from this
#3)#The requirement is that there is a link on the tf tree between the odometry frame and the robot's base_link frame
#A rosbag or bag is a file format in ROS for storing ROS message data. These bags are often created by subscribing to one or more ROS topics, and storing the received message data in an efficient file structure. MATLAB® can read these rosbag files and help with filtering and extracting message data.
#https://microsoft.github.io/AirSim/airsim_ros_pkgs/