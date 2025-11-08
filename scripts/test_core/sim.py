import sys
import numpy as np
from isaacsim import SimulationApp

# 定义机器人与环境路径
# ROBOT_STAGE_PATH = "/World/ArmDog"
# ROBOT_USD_PATH = "assets/ArmDog/armdog_single.usd"
# BACKGROUND_STAGE_PATH = "/World/Robocon2026Map"
# BACKGROUND_USD_PATH = "assets/Map/robocon2026.usd"
ROBOT_STAGE_PATH = "/World/so101_follower"
ROBOT_USD_PATH = "assets/SO101/so101_follower.usd"
BACKGROUND_STAGE_PATH = "/World/background"
BACKGROUND_USD_PATH = "assets/Map/robocon2026.usd"

# CONFIG 字典指定渲染设置，headless=False 表示带 UI 运行仿真
CONFIG = {"renderer": "RaytracedLighting", "headless": False}
simulation_app = SimulationApp(CONFIG)

import carb
import omni.graph.core as og
import usdrt.Sdf
from isaacsim.core.api import SimulationContext
from isaacsim.core.utils import extensions, prims, rotations, stage, viewports
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf

# 启用 ROS2 桥接
extensions.enable_extension("isaacsim.ros2.bridge")
# 初始化仿真 定义世界单位比例（1 单位 = 1 米）
simulation_context = SimulationContext(stage_units_in_meters=1.0)
# 加载环境与机器人
stage.add_reference_to_stage(
    get_assets_root_path() + BACKGROUND_USD_PATH, BACKGROUND_STAGE_PATH
)
prims.create_prim(
    ROBOT_STAGE_PATH,
    "Xform",
    position=np.array([0, -0.64, 0]),
    orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(0, 0, 1), 90)),
    usd_path=ROBOT_USD_PATH,
)

# 创建 ROS2 动作图用于关节控制
og.Controller.edit(
    {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
    {
        "create_nodes": [
            ("OnImpulseEvent", "omni.graph.action.OnImpulseEvent"),
            ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
            ("Context", "isaacsim.ros2.bridge.ROS2Context"),
            ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
            ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
            (
                "ArticulationController",
                "isaacsim.core.nodes.IsaacArticulationController",
            ),
            ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
        ],
        "connect": [
            ("OnImpulseEvent.outputs:execOut", "PublishJointState.inputs:execIn"),
            ("OnImpulseEvent.outputs:execOut", "SubscribeJointState.inputs:execIn"),
            ("OnImpulseEvent.outputs:execOut", "PublishClock.inputs:execIn"),
            ("OnImpulseEvent.outputs:execOut", "ArticulationController.inputs:execIn"),
            ("Context.outputs:context", "PublishJointState.inputs:context"),
            ("Context.outputs:context", "SubscribeJointState.inputs:context"),
            ("Context.outputs:context", "PublishClock.inputs:context"),
            (
                "ReadSimTime.outputs:simulationTime",
                "PublishJointState.inputs:timeStamp",
            ),
            ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
            (
                "SubscribeJointState.outputs:jointNames",
                "ArticulationController.inputs:jointNames",
            ),
            (
                "SubscribeJointState.outputs:positionCommand",
                "ArticulationController.inputs:positionCommand",
            ),
            (
                "SubscribeJointState.outputs:velocityCommand",
                "ArticulationController.inputs:velocityCommand",
            ),
            (
                "SubscribeJointState.outputs:effortCommand",
                "ArticulationController.inputs:effortCommand",
            ),
        ],
        "set_values": [
            ("ArticulationController.inputs:robotPath", ROBOT_STAGE_PATH),
            ("PublishJointState.inputs:topicName", "joint_states_single"),
            ("SubscribeJointState.inputs:topicName", "joint_commands_single"),
            ("PublishJointState.inputs:targetPrim", [usdrt.Sdf.Path(ROBOT_STAGE_PATH)]),
        ],
    },
)

# 启动仿真
simulation_context.initialize_physics()
simulation_context.play()

# 主仿真循环
while simulation_app.is_running():
    simulation_context.step(render=True)
    og.Controller.set(
        og.Controller.attribute("/ActionGraph/OnImpulseEvent.state:enableImpulse"), True
    )

simulation_context.stop()
simulation_app.close()
