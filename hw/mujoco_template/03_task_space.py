"""Task space (operational space) control example.

This example demonstrates how to implement task space control for a robot arm,
allowing it to track desired end-effector positions and orientations. The example
uses a simple PD control law but can be extended to more sophisticated controllers.

Key Concepts Demonstrated:
    - Task space control implementation
    - End-effector pose tracking
    - Real-time target visualization
    - Coordinate frame transformations

Example:
    To run this example:
    
    $ python 03_task_space.py

Notes:
    - The target pose can be modified interactively using the MuJoCo viewer
    - The controller gains may need tuning for different trajectories
    - The example uses a simplified task space controller for demonstration
"""

import numpy as np
from simulator import Simulator
from pathlib import Path
from typing import Dict
import os 
import pinocchio as pin
import matplotlib.pyplot as plt

def plot_results(times: np.ndarray, positions: np.ndarray, velocities: np.ndarray):
    """Plot and save simulation results."""
    # Joint positions plot
    plt.figure(figsize=(10, 6))
    for i in range(positions.shape[1]):
        plt.plot(times, positions[:, i], label=f'Joint {i+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Joint Positions [rad]')
    plt.title('Joint Positions over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/03_task_space_positions.png')
    plt.close()
    
    # Joint velocities plot
    plt.figure(figsize=(10, 6))
    for i in range(velocities.shape[1]):
        plt.plot(times, velocities[:, i], label=f'Joint {i+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Joint Velocities [rad/s]')
    plt.title('Joint Velocities over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/03_task_space_positions_velocities.png')
    plt.close()

def get_trajectory(p_des, t):

    X_des = np.zeros(6)
    dX_des = np.zeros(6)
    ddX_des = np.zeros(6)

    x = p_des[0] + 0.2*np.sin(1*t)
    y = p_des[1] + 0.2*np.cos(1*t)
    z = p_des[2]

    dx = 0.25*np.cos(1*t)
    dy = -0.25*np.sin(1*t)
    dz = 0

    ddx = -0.25*np.sin(1*t)
    ddy = -0.25*np.cos(1*t)
    ddz = 0
    
    X_des[:3] = np.hstack([x,y,z])
    dX_des[:3] = np.hstack([dx,dy,dz])
    ddX_des[:3] = np.hstack([ddx,ddy,ddz])
    
    return X_des, dX_des, ddX_des

def task_space_controller(q: np.ndarray, dq: np.ndarray, t: float, desired: Dict) -> np.ndarray:
    
    kp = np.array([200, 200, 200, 200, 200, 200])*np.eye(6)
    kd = np.array([100, 100, 100, 100, 100, 100])*np.eye(6)

    # Convert desired pose to SE3
    desired_position = desired['pos']
    desired_quaternion = desired['quat'] # [w, x, y, z] in MuJoCo format
    desired_quaternion_pin = np.array([*desired_quaternion[1:], desired_quaternion[0]]) # Convert to [x,y,z,w] for Pinocchio
    # Convert to pose and SE3
    desired_pose = np.concatenate([desired_position, desired_quaternion_pin])
    desired_se3 = pin.XYZQUATToSE3(desired_pose)
    R_des = desired_se3.rotation
    p_des = desired_se3.translation
    
    X_des, dX_des, ddX_des = get_trajectory(p_des, t)

    pin.computeAllTerms(model, data, q, dq)
    pin.forwardKinematics(model, data, q, dq)

    # Get the frame pose
    ee_frame_id = model.getFrameId("end_effector")
    #pin.updateFramePlacement(model, data, ee_frame_id)
    ee_pose = data.oMf[ee_frame_id]
    ee_position = ee_pose.translation
    ee_rotation = ee_pose.rotation

    # Get velocities and accelerations
    frame = pin.WORLD
    twist = pin.getFrameVelocity(model, data, ee_frame_id, frame).vector
    dtwist = pin.getFrameAcceleration(model, data, ee_frame_id, frame)
    J = pin.getFrameJacobian(model, data, ee_frame_id, frame)
    dJ = pin.computeJointJacobiansTimeVariation(model, data, q, dq)


    error_rot = -pin.log3(R_des @ ee_rotation.T)
    X_error = np.zeros(6)
    X_error[:3] = ee_position - X_des[:3]
    X_error[3:] = error_rot

    a_x = ddX_des - kp @ X_error - kd @ (twist - dX_des)
    a_q = np.linalg.inv(J)@(a_x - dJ@dq)

    tau = data.M @ a_q + data.nle

    print(f'X_error : {X_error}')

    return tau

def main():
    # Create logging directories
    Path("logs/videos").mkdir(parents=True, exist_ok=True)
    
    print("\nRunning task space controller...")
    sim = Simulator(
        xml_path="robots/universal_robots_ur5e/scene.xml",
        enable_task_space=True,
        show_viewer=True,
        record_video=True,
        video_path="logs/videos/03_task_space.mp4",
        fps=30,
        width=1920,
        height=1080
    )
    sim.set_controller(task_space_controller)
    sim.run(time_limit=20.0)

    # Process and save results
    times = np.array(sim.times)
    positions = np.array(sim.positions)
    velocities = np.array(sim.velocities)
    plot_results(times, positions, velocities)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "robots/universal_robots_ur5e/ur5e.xml")
    model = pin.buildModelFromMJCF(xml_path)
    data = model.createData()
    main() 