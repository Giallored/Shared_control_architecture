// ROS headers
#include <ros/ros.h>

// MoveIt! headers
#include <moveit/move_group_interface/move_group_interface.h>

// Std C++ headers
#include <string>
#include <vector>
#include <map>
#include <geometry_msgs/Pose.h>


int main(int argc, char** argv)
{
    ros::init(argc, argv, "plan_arm_torso_ik");
    ros::NodeHandle nh;
    ros::Subscriber sub = n.subscribe("usr_cmd/pose", 1000, callback);
    ros::spin();
  
  return 0;
}


void callback(const geometry_msgs::Pose::ConstPtr& pose)
{
  std::vector<std::string> torso_arm_joint_names;
  //select group of joints
  moveit::planning_interface::MoveGroup group_arm_torso("arm_torso");
  //choose your preferred planner
  group_arm_torso.setPlannerId("SBLkConfigDefault");
  group_arm_torso.setPoseReferenceFrame("base_footprint");
  group_arm_torso.setPoseTarget(goal_pose);

  ROS_INFO_STREAM("Planning to move " <<
                  group_arm_torso.getEndEffectorLink() << " to a target pose expressed in " <<
                  group_arm_torso.getPlanningFrame());

  group_arm_torso.setStartStateToCurrentState();
  group_arm_torso.setMaxVelocityScalingFactor(1.0);


  moveit::planning_interface::MoveGroup::Plan my_plan;
  //set maximum time to find a plan
  group_arm_torso.setPlanningTime(5.0);
  bool success = group_arm_torso.plan(my_plan);

  if ( !success )
    throw std::runtime_error("No plan found");

  ROS_INFO_STREAM("Plan found in " << my_plan.planning_time_ << " seconds");

  // Execute the plan
  ros::Time start = ros::Time::now();

  group_arm_torso.move();

  ROS_INFO_STREAM("Motion duration: " << (ros::Time::now() - start).toSec());

  spinner.stop();

}

