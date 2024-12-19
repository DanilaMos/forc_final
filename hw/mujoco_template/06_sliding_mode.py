import numpy as np
from simulator import Simulator
from pathlib import Path
import os
from typing import Dict
import matplotlib.pyplot as plt
import pinocchio as pin

q_des = np.array([-1.4, -1.2, 1., -0.2, 0.2, 0.5])

def plot_results(times: np.ndarray, positions: np.ndarray, velocities: np.ndarray):
    """Plot and save simulation results."""

    colors = ['b', 'g', 'r', 'c', 'm', 'y']

    # Joint positions plot
    plt.figure(figsize=(10, 6))
    for i in range(positions.shape[1]):
        plt.plot(times, positions[:, i], colors[i], label=f'Joint {i+1}')
        plt.hlines(q_des[i], 0, times[-1], colors[i], '--')  
    plt.xlabel('Time [s]')
    plt.ylabel('Joint Positions [rad]')
    plt.title('Joint Positions over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/06_positions_sliding.png')
    plt.close()

    # Joint errors plot
    plt.figure(figsize=(10, 6))
    for i in range(positions.shape[1]):
        pos_error = np.array([q_des[i] for j in range(positions.shape[0])])
        plt.plot(times, positions[:, i] - pos_error, label=f'Joint {i+1}') 
    plt.xlabel('Time [s]')
    plt.ylabel('Joint Positions Errors [rad]')
    plt.title('Joint Positions Errors over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/06_errors_sliding.png')
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
    plt.savefig('logs/plots/06_velocities_sliding.png')
    plt.close()

def u_s(r,s,p):
    """Compute saturation function"""
    if np.linalg.norm(s, ord=2) > p:
        u = r * s / np.linalg.norm(s, ord=2)
    else:
        u = r * s / p
    return u


def joint_controller(q: np.ndarray, dq: np.ndarray, t: float) -> np.ndarray:
    """Joint space PD controller.
    
    Args:
        q: Current joint positions [rad]
        dq: Current joint velocities [rad/s]
        t: Current simulation time [s]
        
    Returns:
        tau: Joint torques command [Nm]
    """
    
    # Target joint configuration
    q0 = q_des
    dq0 = np.zeros(6)
    ddq0 = np.zeros(6)

    # Errors compute
    q_err = q0 - q
    dq_err = dq0 - dq
    
    # Dynamics computation
    pin.computeAllTerms(model, data, q, dq)
    M = data.M
    nle = data.nle

    # Control parameters
    L = np.diag([400, 400, 400, 100, 100, 1])
    k = 300
    p = 100 

    # Sliding surface
    s = dq_err + L @ q_err

    # Compute the control law
    u_hat = M @ (ddq0 + L@dq_err) + nle
    tau = u_hat + u_s(k, s, p)
    return tau

def main():
    # Create logging directories
    Path("logs/videos").mkdir(parents=True, exist_ok=True)
    
    # Initialize simulator
    sim = Simulator(
        xml_path="robots/universal_robots_ur5e/scene.xml",
        enable_task_space=False,  # Using joint space control
        show_viewer=False,
        record_video=True,
        video_path="logs/videos/06_sliding_mode.mp4",
        fps=30,
        width=1920,
        height=1080
    )
    
    # Set joint damping (example values, adjust as needed)
    damping = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1])  # Nm/rad/s
    sim.set_joint_damping(damping)
    
    # Set joint friction (example values, adjust as needed)
    friction = np.array([1.5, 0.5, 0.5, 0.1, 0.1, 0.1])  # Nm
    sim.set_joint_friction(friction)
    
    # Get original properties
    ee_name = "end_effector"
    
    original_props = sim.get_body_properties(ee_name)
    print(f"\nOriginal end-effector properties:")
    print(f"Mass: {original_props['mass']:.3f} kg")
    print(f"Inertia:\n{original_props['inertia']}")
    
    # Add the end-effector mass and inertia
    sim.modify_body_properties(ee_name, mass=0)
    # Print modified properties
    props = sim.get_body_properties(ee_name)
    print(f"\nModified end-effector properties:")
    print(f"Mass: {props['mass']:.3f} kg")
    print(f"Inertia:\n{props['inertia']}")
    
    # Set controller and run simulation
    sim.set_controller(joint_controller)
    sim.run(time_limit=10.0)

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