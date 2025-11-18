#!/usr/bin/env python3
import rclpy
import rclpy.parameter
import torch
import numpy as np
import io
import time
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Imu
from message_filters import Subscriber, TimeSynchronizer

class Go2Controller(Node):
    def __init__(self):
        super().__init__('go2_controller')

        self.declare_parameter('policy_path', 'policy/policy_1/exported/policy.pt')
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        self._logger = self.get_logger()

        sim_qos_profile = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.QoSDurabilityPolicy.VOLATILE,
            history=rclpy.qos.QoSHistoryPolicy.KEEP_ALL
        )

        self._cmd_vel_subscription = self.create_subscription(Twist, 'cmd_vel', self.cmd_vel_callback, 10)
        self._joint_publisher = self.create_publisher(JointState, f'joint_command', sim_qos_profile)
        self._imu_sub_filter = Subscriber(self, Imu, f'imu', qos_profile=sim_qos_profile)
        self._joint_states_sub_filter = Subscriber(self, JointState, f'joint_states', qos_profile=sim_qos_profile)

        queue_size = 10
        subscribers = [self._joint_states_sub_filter, self._imu_sub_filter]
        self._sync = TimeSynchronizer(subscribers, queue_size)
        self._sync.registerCallback(self.synchronized_callback)

        self.policy_path = self.get_parameter('policy_path').get_parameter_value().string_value
        self.load_policy()

        self._joint_state = JointState()
        self._joint_command = JointState()
        self._cmd_vel = Twist()
        self._imu = Imu()

        self._policy_counter = 0
        self._decimation = 1
        self._last_tick_time = self.get_clock().now().nanoseconds * 1e-9
        self.base_lin_vel = np.zeros(3)
        self._dt = 0.0

        self.default_pos = np.array([
            0.1, 0.1, -0.1, -0.1, 
            0.8, 0.8, 1.0, 1.0,
            -1.5, -1.5, -1.5, -1.5,
        ])
        self.joint_names = [
            "FL_hip_joint",
            "RL_hip_joint",
            "FR_hip_joint",
            "RR_hip_joint",
            "FL_thigh_joint",
            "FR_thigh_joint",
            "RL_thigh_joint",
            "RR_thigh_joint",
            "FL_calf_joint",
            "FR_calf_joint",
            "RL_calf_joint",
            "RR_calf_joint",
        ]
        self._action_scale = [
            0.125, 0.125, 0.125, 0.125, 
            0.25, 0.25, 0.25, 0.25, 
            0.25, 0.25, 0.25, 0.25
        ]
        
        self.action_length = len(self.default_pos)
        self._previous_action = np.zeros(self.action_length)

        self._filter_alpha = 0.99
        self._filtered_action = np.zeros(self.action_length)
        self._filter_pre_action = np.zeros(self.action_length)

        self._logger.info("Go2Controller initialized")

    def load_policy(self):
        # Load policy from file to io.BytesIO object
        self._logger.info(f"Loading policy from {self.policy_path}")
        with open(self.policy_path, "rb") as f:
            buffer = io.BytesIO(f.read())
        # Load TorchScript model from buffer
        self.policy = torch.jit.load(buffer)

    def cmd_vel_callback(self, msg):
        self._cmd_vel = msg
        self._logger.info(f"Received cmd_vel: {msg.linear.x}, {msg.linear.y}, {msg.angular.z}")

    def synchronized_callback(self, joint_state: JointState, imu):
        # Reset if time jumped backwards (most likely due to sim time reset)
        now = self.get_clock().now().nanoseconds * 1e-9
        if now < self._last_tick_time:
            self._logger.error(
                f"{self._get_stamp_prefix()} Time jumped backwards. Resetting."
            )

        # Calculate time delta since last tick
        self._dt = now - self._last_tick_time
        self._last_tick_time = now

        # Run the control policy
        # self.forward(joint_state, imu)
        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(joint_state, imu)
            self.action = self._compute_action(obs)

            self._filtered_action = self._filter_alpha * self.action + (1 - self._filter_alpha) * self._filter_pre_action
            self._filter_pre_action = self._filtered_action.copy()
            self._policy_counter = 0
        else:
            self._policy_counter += 1
            return

        # Prepare and publish the joint command message
        self._joint_command.header.stamp = self.get_clock().now().to_msg()
        self._joint_command.name = self.joint_names

        # Compute final joint positions by adding scaled actions to default positions
        action_pos = self.default_pos + np.clip(self._filtered_action, -100.0, 100.0) * self._action_scale
        # action_pos[8:10], action_pos[14:24] = np.zeros(2), np.zeros(10)
        self._joint_command.position = action_pos.tolist()
        self._joint_command.velocity = np.zeros(len(self.joint_names)).tolist()
        self._joint_command.effort = np.zeros(len(self.joint_names)).tolist()
        self._joint_publisher.publish(self._joint_command)

    def _get_stamp_prefix(self) -> str:
        now = time.time()
        now_ros = self.get_clock().now().nanoseconds / 1e9
        return f"[{now}][{now_ros}]"

    def header_time_in_seconds(self, header) -> float:
        return header.stamp.sec + header.stamp.nanosec * 1e-9

    def _compute_observation(self, joint_state: JointState, imu: Imu):
        # Extract quaternion orientation from IMU
        quat_I = imu.orientation
        quat_array = np.array([quat_I.w, quat_I.x, quat_I.y, quat_I.z])

        # Convert quaternion to rotation matrix
        # (transpose for body to inertial frame)
        R_BI = self.quat_to_rot_matrix(quat_array).T
        projected_gravity = np.matmul(R_BI, [0, 0, -1.0])

        # Extract linear acceleration and integrate to estimate velocity
        lin_acc_b = np.array(
            [
                imu.linear_acceleration.x,
                imu.linear_acceleration.y,
                imu.linear_acceleration.z,
            ]
        )

        # Simple integration to estimate velocity
        self.base_lin_vel = lin_acc_b * self._dt + self.base_lin_vel

        # Extract angular velocity
        base_ang_vel = np.array(
            [imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z]
        )

        # Initialize observation vector
        obs = np.zeros(
            # 3
            + 3
            + 3
            + 3
            + self.action_length
            + self.action_length
            + self.action_length
        )

        idx = 0
        # Fill observation vector components:
        # Base linear velocity (3)
        # obs[idx : idx + 3] = self.base_lin_vel
        # idx += 3

        # Base angular velocity (3)
        obs[idx : idx + 3] = np.clip(base_ang_vel * 0.25, -100.0, 100.0)
        idx += 3

        # # Base linear acceleration (3)
        # obs[idx : idx + 3] = lin_acc_b
        # idx += 3

        # Gravity direction (3)
        obs[idx:idx+3] = np.clip(projected_gravity * 1.0, -100.0, 100.0)
        idx += 3

        # Velocity commands (3)
        velocity_commands = [
            self._cmd_vel.linear.x,
            self._cmd_vel.linear.y,
            self._cmd_vel.angular.z,
        ]
        obs[idx:idx+3] = np.clip(np.array(velocity_commands) * 1.0, -100.0, 100.0)
        idx += 3

        # Joint states (19 positions + 19 velocities)
        joint_pos = np.zeros(self.action_length)
        joint_vel = np.zeros(self.action_length)

        # Map joint states from message to our ordered arrays
        for i, name in enumerate(self.joint_names):
            if name in joint_state.name:
                joint_idx = joint_state.name.index(name)
                joint_pos[i] = joint_state.position[joint_idx]
                joint_vel[i] = joint_state.velocity[joint_idx]

        # joint_pos[-4:] = np.zeros(4)

        # Store joint positions relative to default pose
        obs[idx : idx + self.action_length] = np.clip((joint_pos - self.default_pos) * 1.0, -100.0, 100.0)
        idx += self.action_length

        # Store joint velocities
        obs[idx : idx + self.action_length] = np.clip(joint_vel * 0.05, -100.0, 100.0)
        idx += self.action_length

        # Store previous actions
        obs[idx : idx + self.action_length] = np.clip(self._previous_action * 1.0, -100.0, 100.0)
        idx += self.action_length

        return obs

    def _compute_action(self, obs):
        # Run inference with the PyTorch policy
        with torch.no_grad():
            obs = torch.from_numpy(obs).view(1, -1).float()
            # print(obs.shape)
            action = self.policy(obs).detach().view(-1).numpy()
            # print(action.shape)
            # print(action)
        return action

    def forward(self, joint_state: JointState, imu: Imu):
        # Compute observation from current state
        obs = self._compute_observation(joint_state, imu)

        # Run policy at reduced frequency (every _decimation ticks)
        if self._policy_counter % self._decimation == 0:
            self.action = self._compute_action(obs)
            self._previous_action = self.action.copy()
        self._policy_counter += 1

    def quat_to_rot_matrix(self, quat: np.ndarray) -> np.ndarray:
        q = np.array(quat, dtype=np.float64, copy=True)
        nq = np.dot(q, q)
        if nq < 1e-10:
            return np.identity(3)
        q *= np.sqrt(2.0 / nq)
        q = np.outer(q, q)
        return np.array(
            (
                (1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]),
                (q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]),
                (q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]),
            ),
            dtype=np.float64,
        )


def main(args=None):
    rclpy.init(args=args)
    node = Go2Controller()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
