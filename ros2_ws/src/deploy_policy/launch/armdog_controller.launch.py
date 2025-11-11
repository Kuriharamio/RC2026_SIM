from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    policy_path = os.path.join(
        get_package_share_directory('deploy_policy'),
        'policy/himloco/exported/policy.pt'
    )
    return LaunchDescription([
        DeclareLaunchArgument(
            "publish_period_ms",
            default_value="5",
            description="publishing dt in milliseconds"),
        DeclareLaunchArgument(
            "policy_path",
            default_value=policy_path,
            description="path to the policy file"),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="True",
            description="Use simulation (Omniverse Isaac Sim) clock if true"),
        DeclareLaunchArgument(
            "dog_type",
            default_value="none",
            description="Dog type: none or single or dual"),
        Node(
            package='deploy_policy',
            executable='armdog_controller.py',
            name='armdog_controller',
            output="screen",
            parameters=[{
                'publish_period_ms': LaunchConfiguration('publish_period_ms'),
                'policy_path': LaunchConfiguration('policy_path'),
                "use_sim_time": LaunchConfiguration('use_sim_time'),
                "dog_type": LaunchConfiguration('dog_type'),
            }]
            
        ),
    ])
