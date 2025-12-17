import numpy as np
import matplotlib.pyplot as plt
import time
import math
import pybullet as p
import pybullet_data
import os

USE_GUI = True
TOTAL_STEPS = 1200 

def run_final_show():
    try:
        p.disconnect()
    except:
        pass
        
    p.connect(p.GUI if USE_GUI else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1./240.)

    p.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=45, cameraPitch=-35, cameraTargetPosition=[0,0,0])
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0) 

    plane_id = p.loadURDF("plane.urdf")
    p.changeDynamics(plane_id, -1, lateralFriction=0.5) 

    my_urdf_path = os.path.join(os.getcwd(), "Capsule_robot_description", "urdf", "Capsule_robot.urdf")
    start_pos = [0, 0, 0.05] 
    robot_id = p.loadURDF(my_urdf_path, start_pos, useFixedBase=False)

    joints = []
    num_joints = p.getNumJoints(robot_id)
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        if info[2] != p.JOINT_FIXED:
            joints.append(i)
            p.changeDynamics(robot_id, i, jointDamping=0.0)

    for i in range(-1, num_joints):
        p.changeDynamics(robot_id, i, lateralFriction=0.5, restitution=0)

    print(f"--- 🎬 발표용 데모 및 데이터 수집 시작 (Steps: {TOTAL_STEPS}) ---")
    
    log_actions = []      
    log_velocity = []    
    log_position = []    
    

    freq = 8.0      
    wave_len = 1.0   
    amp = 0.8        


    for t_step in range(TOTAL_STEPS):
        t = t_step * (1./240.)
        
        current_actions = []
        
        for i, joint_idx in enumerate(joints):
            raw_signal = math.sin(t * freq - i * wave_len)
            target_angle = amp * raw_signal
            
            laser_val = (raw_signal + 1) / 2
            current_actions.append(laser_val)
            
            p.setJointMotorControl2(
                robot_id, joint_idx, 
                controlMode=p.POSITION_CONTROL, 
                targetPosition=target_angle, 
                force=500.0,
                maxVelocity=10.0
            )

        lin_vel, _ = p.getBaseVelocity(robot_id)
        pos, _ = p.getBasePositionAndOrientation(robot_id)
        
        log_actions.append(current_actions)
        log_velocity.append(lin_vel[0]) 
        log_position.append([pos[0], pos[1]]) 

        p.stepSimulation()
        time.sleep(1./480.) 
        
        p.resetDebugVisualizerCamera(0.8, 45, -35, pos)

    print("--- 주행 종료. 데이터 분석 그래프를 생성합니다... ---")
    p.disconnect()
    
    action_data = np.array(log_actions).T # (Joints, Time)
    vel_data = np.array(log_velocity)
    pos_data = np.array(log_position)
    time_axis = np.arange(TOTAL_STEPS) * (1./240.)

    plt.style.use('default') 

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Biomimetic Robot Performance Analysis (Frequency: {freq}Hz)", fontsize=16, fontweight='bold')


    ax1 = axes[0, 0]
    im = ax1.imshow(action_data, aspect='auto', cmap='magma', interpolation='bilinear',
                    extent=[0, time_axis[-1], num_joints-1, 0])
    ax1.set_title("(A) Spatiotemporal Gait Pattern", fontweight='bold')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Joint Index (Head -> Tail)")
    ax1.set_yticks(np.arange(num_joints))
    ax1.set_yticklabels([f'J{i}' for i in range(num_joints)])
    fig.colorbar(im, ax=ax1, label="Action Intensity (0~1)")

    ax1.arrow(0.5, 0, 1.0, 4, head_width=0.1, head_length=0.5, fc='cyan', ec='cyan', linewidth=2)
    ax1.text(0.6, 2, "Wave Propagation", color='cyan', fontsize=10, fontweight='bold', rotation=75)

    ax2 = axes[0, 1]
    ax2.plot(time_axis, vel_data, color='#1f77b4', linewidth=1.5, alpha=0.8)
    ax2.axhline(y=np.mean(vel_data), color='red', linestyle='--', label=f'Avg Speed: {np.mean(vel_data):.3f} m/s')
    ax2.set_title("(B) Forward Velocity Profile", fontweight='bold')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Velocity (m/s)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)


    ax3 = axes[1, 0]
    ax3.plot(pos_data[:, 0], pos_data[:, 1], color='purple', linewidth=2)
    ax3.scatter(pos_data[0, 0], pos_data[0, 1], color='green', label='Start', zorder=5)
    ax3.scatter(pos_data[-1, 0], pos_data[-1, 1], color='red', label='End', zorder=5)
    ax3.set_title("(C) Robot Trajectory (Top-View)", fontweight='bold')
    ax3.set_xlabel("X Position (m)")
    ax3.set_ylabel("Y Position (m)")
    ax3.axis('equal') 
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    ax4 = axes[1, 1]
    zoom_range = slice(100, 300) 
    ax4.plot(time_axis[zoom_range], action_data[0, zoom_range], label='Head (J0)', color='red', linestyle='-')
    ax4.plot(time_axis[zoom_range], action_data[4, zoom_range], label='Mid (J4)', color='green', linestyle='--')
    ax4.plot(time_axis[zoom_range], action_data[8, zoom_range], label='Tail (J8)', color='blue', linestyle='-.')
    ax4.set_title("(D) Phase Lag Verification", fontweight='bold')
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Joint Action")
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.text(time_axis[150], 0.8, "Time Delay confirms\nTraveling Wave", fontsize=9, bbox=dict(facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_final_show()
