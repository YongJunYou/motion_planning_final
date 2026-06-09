아이작심 버전 5.0.0

아이작랩 버전 2.2.1

<img width="724" height="63" alt="image" src="https://github.com/user-attachments/assets/9c707eec-95dc-4fed-a61a-b77ccea80033" />

실행명령어 (conda 사용중)
(env_isaacsim) yyj@larr-yyj:~/IsaacLab/IsaacLab$ ./isaaclab.sh -p ~/motion_planning_final/main.py --mode hover

토픽 안뜰경우
unset PYTHONPATH
unset AMENT_PREFIX_PATH
unset COLCON_PREFIX_PATH

export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/humble/lib
