from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    policy_path = os.path.join(
        get_package_share_directory('deploy_policy'),
        'policy/go2w/rough/exported/policy.pt'
    )
    return LaunchDescription([
        DeclareLaunchArgument(
            "policy_path",
            default_value=policy_path,
            description="path to the policy file"),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="True",
            description="Use simulation (Omniverse Isaac Sim) clock if true"),
        Node(
            package='deploy_policy',
            executable='go2w_controller.py',
            name='go2w_controller',
            output="screen",
            parameters=[{
                'policy_path': LaunchConfiguration('policy_path'),
                "use_sim_time": LaunchConfiguration('use_sim_time'),
            }]
            
        ),
    ])
