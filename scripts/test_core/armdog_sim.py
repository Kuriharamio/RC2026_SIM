from isaacsim import SimulationApp

CONFIG = {"renderer": "RaytracedLighting", "headless": False}
simulation_app = SimulationApp(CONFIG)

from isaacsim.core.api import SimulationContext

simulation_context = SimulationContext(stage_units_in_meters=1.0, physics_dt=1/200.0, rendering_dt=1/30.0)

from isaacsim.core.utils.prims import define_prim
from pxr import PhysicsSchemaTools

prim = define_prim("/World", "Xform")
prim.GetReferences().AddReference("assets/Simulation/sim.usd")

PhysicsSchemaTools.addDomeLight

simulation_context.initialize_physics()
simulation_context.play()

while simulation_app.is_running():
    simulation_context.step(render=True)

simulation_context.stop()
simulation_app.close()
