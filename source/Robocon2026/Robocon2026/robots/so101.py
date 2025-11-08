import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

"""Configuration for the SO101 Follower Robot."""
SO101_FOLLOWER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="assets/SO101/so101.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(2.2, -0.61, 0.89),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.8,
            "wrist_flex": -0.8,
            "wrist_roll": 1.57,
            "gripper": 0.3,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "sts3215-gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper"],
            effort_limit_sim=15,
            velocity_limit_sim=2.5,
            stiffness=60.0,
            damping=20.0,
            # damping=5.0,
        ),
        # "sts3215-arm": ImplicitActuatorCfg(
        #     joint_names_expr=[
        #         "shoulder_pan",
        #         "shoulder_lift",
        #         "elbow_flex",
        #         "wrist_flex",
        #         "wrist_roll",
        #     ],
        #     effort_limit_sim=10,
        #     velocity_limit_sim=10,
        #     stiffness=17.8,
        #     damping=0.60,
        # ),
        "sts3215-arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
            ],
            effort_limit_sim=1.9,
            velocity_limit_sim=1.5,
            stiffness={
                "shoulder_pan": 200.0,  # Highest - moves all mass
                "shoulder_lift": 170.0,  # Slightly less than rotation
                "elbow_flex": 120.0,  # Reduced based on less mass
                "wrist_flex": 80.0,  # Reduced for less mass
                "wrist_roll": 50.0,  # Low mass to move
            },
            damping={
                "shoulder_pan": 80.0,
                "shoulder_lift": 65.0,
                "elbow_flex": 45.0,
                "wrist_flex": 30.0,
                "wrist_roll": 20.0,
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

# joint limit written in USD (degree)
SO101_FOLLOWER_USD_JOINT_LIMLITS = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 90.0),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-160.0, 160.0),
    "gripper": (-10, 100.0),
}

# motor limit written in real device (normalized to related range)
SO101_FOLLOWER_MOTOR_LIMITS = {
    'shoulder_pan': (-100.0, 100.0),
    'shoulder_lift': (-100.0, 100.0),
    'elbow_flex': (-100.0, 100.0),
    'wrist_flex': (-100.0, 100.0),
    'wrist_roll': (-100.0, 100.0),
    'gripper': (0.0, 100.0),
}


SO101_FOLLOWER_REST_POSE_RANGE = {
    "shoulder_pan": (0 - 30.0, 0 + 30.0),  # 0 degree
    "shoulder_lift": (-100.0 - 30.0, -100.0 + 30.0),  # -100 degree
    "elbow_flex": (90.0 - 30.0, 90.0 + 30.0),  # 90 degree
    "wrist_flex": (50.0 - 30.0, 50.0 + 30.0),  # 50 degree
    "wrist_roll": (0.0 - 30.0, 0.0 + 30.0),  # 0 degree
    "gripper": (-10.0 - 30.0, -10.0 + 30.0),  # -10 degree
}
