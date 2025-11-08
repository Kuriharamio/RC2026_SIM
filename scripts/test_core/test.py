# launch Isaac Sim before any other imports
# default first two lines in any standalone application
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})  # we can also run as headless.

from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
import numpy as np

world = World()
world.scene.add_default_ground_plane()

add_reference_to_stage(usd_path="assets/Simulation/sim.usd", prim_path="/World/Sim")  # add robot to stage
# armdog = Articulation(prim_paths_expr="/World/Armdog", name="armdog")  # create an articulation object
# armdog.set_joint_positions([[0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, 0.0, -1.5, -1.5, -1.5, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0]])
# armdog.set_world_poses(positions=np.array([[0.0, 1.0, 0.4]]) / get_stage_units())

world.reset()
for i in range(500):
    # armdog.set_joint_positions([[0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, 0.0, -1.5, -1.5, -1.5, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0]])
    world.step(render=True)  # execute one physics step and one rendering step

simulation_app.close()  # close Isaac Sim
