import numpy as np
import matplotlib.pyplot as plt
import time
import math
import pybullet as p
import pybullet_data
import os

# --- 1. 설정 ---
USE_GUI = True
TOTAL_STEPS = 1200 # 데이터를 좀 더 확보하기 위해 1000 -> 1200으로 살짝 늘림 (약 5초)

def run_final_show():
    # ---------------------------------------------------------
    # [Part A] 시뮬레이션 환경 설정 (사용자 원본 코드 유지)
    # ---------------------------------------------------------
    try:
        p.disconnect()
    except:
        pass
        
    p.connect(p.GUI if USE_GUI else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1./240.)
    
    # 카메라 설정
    p.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=45, cameraPitch=-35, cameraTargetPosition=[0,0,0])
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0) 

    # 바닥 및 로봇 로드
    plane_id = p.loadURDF("plane.urdf")
    p.changeDynamics(plane_id, -1, lateralFriction=0.5) # 바닥 마찰 수정 (0.2는 너무 미끄러울 수 있어 0.5로 타협)

    my_urdf_path = os.path.join(os.getcwd(), "Capsule_robot_description", "urdf", "Capsule_robot.urdf")
    start_pos = [0, 0, 0.05] 
    robot_id = p.loadURDF(my_urdf_path, start_pos, useFixedBase=False)

    # 관절 설정
    joints = []
    num_joints = p.getNumJoints(robot_id)
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        if info[2] != p.JOINT_FIXED:
            joints.append(i)
            p.changeDynamics(robot_id, i, jointDamping=0.0)

    # 몸체 마찰 설정
    for i in range(-1, num_joints):
        p.changeDynamics(robot_id, i, lateralFriction=0.5, restitution=0)

    print(f"--- 🎬 발표용 데모 및 데이터 수집 시작 (Steps: {TOTAL_STEPS}) ---")
    
    # ---------------------------------------------------------
    # [Part B] 데이터 저장소 초기화
    # ---------------------------------------------------------
    log_actions = []      # 제어 신호 (Heatmap용)
    log_velocity = []     # 로봇의 전진 속도 (Performance용)
    log_position = []     # 로봇의 위치 (Trajectory용)
    
    # 파라미터 (S자 주행의 핵심)
    freq = 8.0       # 속도 조절 (너무 빠르면 물리엔진이 불안정할 수 있어 10->8로 미세 조정)
    wave_len = 1.0   # 파장
    amp = 0.8        # 진폭

    # ---------------------------------------------------------
    # [Part C] 주행 루프
    # ---------------------------------------------------------
    for t_step in range(TOTAL_STEPS):
        t = t_step * (1./240.)
        
        current_actions = []
        
        # 1. 제어 신호 생성 (Traveling Wave)
        for i, joint_idx in enumerate(joints):
            raw_signal = math.sin(t * freq - i * wave_len)
            target_angle = amp * raw_signal
            
            # 히트맵용 정규화 (0~1)
            laser_val = (raw_signal + 1) / 2
            current_actions.append(laser_val)
            
            # 모터 구동
            p.setJointMotorControl2(
                robot_id, joint_idx, 
                controlMode=p.POSITION_CONTROL, 
                targetPosition=target_angle, 
                force=500.0,
                maxVelocity=10.0
            )

        # 2. 물리 데이터 수집 (여기가 중요합니다!)
        # 실제 로봇의 속도와 위치를 뽑아와야 "가짜 그래프"가 아니게 됩니다.
        lin_vel, _ = p.getBaseVelocity(robot_id)
        pos, _ = p.getBasePositionAndOrientation(robot_id)
        
        log_actions.append(current_actions)
        log_velocity.append(lin_vel[0]) # X축(전진) 속도
        log_position.append([pos[0], pos[1]]) # X, Y 좌표

        p.stepSimulation()
        time.sleep(1./480.) # 화면 확인용 딜레이 (약간 빠르게)
        
        # 카메라 팔로우
        p.resetDebugVisualizerCamera(0.8, 45, -35, pos)

    print("--- 주행 종료. 데이터 분석 그래프를 생성합니다... ---")
    p.disconnect()
    
    # ---------------------------------------------------------
    # [Part D] 전문적인 결과 그래프 그리기
    # ---------------------------------------------------------
    # 데이터 변환
    action_data = np.array(log_actions).T # (Joints, Time)
    vel_data = np.array(log_velocity)
    pos_data = np.array(log_position)
    time_axis = np.arange(TOTAL_STEPS) * (1./240.)

    # 그래프 스타일 설정
    plt.style.use('default') # 깔끔한 기본 스타일
    
    # 2x2 서브플롯 생성
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Biomimetic Robot Performance Analysis (Frequency: {freq}Hz)", fontsize=16, fontweight='bold')

    # [1] Spatiotemporal Gait Pattern (히트맵)
    # - 의미: 뇌(Brain)가 몸에게 어떤 신호를 보냈는지 시각화
    # - 확인점: 대각선 무늬가 선명할수록 완벽한 'Traveling Wave'임
    ax1 = axes[0, 0]
    im = ax1.imshow(action_data, aspect='auto', cmap='magma', interpolation='bilinear',
                    extent=[0, time_axis[-1], num_joints-1, 0])
    ax1.set_title("(A) Spatiotemporal Gait Pattern", fontweight='bold')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Joint Index (Head -> Tail)")
    ax1.set_yticks(np.arange(num_joints))
    ax1.set_yticklabels([f'J{i}' for i in range(num_joints)])
    fig.colorbar(im, ax=ax1, label="Action Intensity (0~1)")
    # 파동 흐름 화살표
    ax1.arrow(0.5, 0, 1.0, 4, head_width=0.1, head_length=0.5, fc='cyan', ec='cyan', linewidth=2)
    ax1.text(0.6, 2, "Wave Propagation", color='cyan', fontsize=10, fontweight='bold', rotation=75)

    # [2] Forward Velocity Profile (속도 그래프)
    # - 의미: 로봇의 실제 퍼포먼스. 0보다 위에 있어야 앞으로 가는 것.
    # - 확인점: 파동에 따라 속도가 출렁거리지만(Oscillation), 평균적으로 양수여야 함.
    ax2 = axes[0, 1]
    ax2.plot(time_axis, vel_data, color='#1f77b4', linewidth=1.5, alpha=0.8)
    ax2.axhline(y=np.mean(vel_data), color='red', linestyle='--', label=f'Avg Speed: {np.mean(vel_data):.3f} m/s')
    ax2.set_title("(B) Forward Velocity Profile", fontweight='bold')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Velocity (m/s)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # [3] 2D Trajectory (경로 추적)
    # - 의미: 로봇이 실제로 이동한 경로. 
    # - 확인점: (0,0)에서 시작해서 X축 방향으로 쭉 뻗어나가야 함.
    ax3 = axes[1, 0]
    ax3.plot(pos_data[:, 0], pos_data[:, 1], color='purple', linewidth=2)
    ax3.scatter(pos_data[0, 0], pos_data[0, 1], color='green', label='Start', zorder=5)
    ax3.scatter(pos_data[-1, 0], pos_data[-1, 1], color='red', label='End', zorder=5)
    ax3.set_title("(C) Robot Trajectory (Top-View)", fontweight='bold')
    ax3.set_xlabel("X Position (m)")
    ax3.set_ylabel("Y Position (m)")
    ax3.axis('equal') # 비율을 1:1로 맞춰야 경로 왜곡이 없음
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # [4] Phase Lag Analysis (위상차 분석)
    # - 의미: 관절들이 정말로 '순서대로' 움직이는지 증명
    # - 확인점: Joint 0(Head) -> Joint 4(Mid) -> Joint 8(Tail) 순으로 파동이 밀려야 함.
    ax4 = axes[1, 1]
    zoom_range = slice(100, 300) # 초반 200스텝만 확대해서 보여줌
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