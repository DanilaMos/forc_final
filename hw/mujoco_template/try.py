import mujoco
import pinocchio as pin
import os
import numpy as np

#model = mujoco.MjModel.from_xml_path("robots/universal_robots_ur5e/scene.xml")

current_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(current_dir, "robots/universal_robots_ur5e/ur5e.xml")
model = pin.buildModelFromMJCF(xml_path)
data = model.createData()

q = np.random.uniform(-np.pi, np.pi, model.nq)  # random joint positions
dq = np.random.uniform(-1, 1, model.nv)  # random joint velocities
ddq = np.random.uniform(-1, 1, model.nv)  # random joint accelerations

regressor = pin.computeJointTorqueRegressor(model, data, q, dq, ddq)

print(model.frames[-1])
