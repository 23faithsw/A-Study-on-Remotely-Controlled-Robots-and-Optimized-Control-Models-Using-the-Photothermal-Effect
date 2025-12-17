import gymnasium as gym
import pybullet as p
import pybullet_data
import numpy as np
import os
import math

class LCP_CaterpillarEnv(gym.Env):
    def __init__(self, render=False):
        try: p.disconnect()
        except: pass
        
        if render:
            p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
            p.resetDebugVisualizerCamera(1.0, 0, -40, [0,0,0])
        else:
            p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1./240.)

        plane_id = p.loadURDF("plane.urdf")
        p.changeDynamics(plane_id, -1, lateralFriction=0.5) 

        self.start_pos = [0, 0, 0.05]
        self.start_orn = p.getQuaternionFromEuler([0, 0, 0])
        
        my_urdf = os.path.join(os.getcwd(), "Capsule_robot_description", "urdf", "Capsule_robot.urdf")
        self.robot_id = p.loadURDF(my_urdf, self.start_pos, self.start_orn, useFixedBase=False)
        
        self.joints = []
        for i in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, i)
            if info[2] != p.JOINT_FIXED:
                self.joints.append(i)
                p.changeDynamics(self.robot_id, i, linearDamping=0.0, angularDamping=0.0)
        
        self.num_joints = len(self.joints)

        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.num_joints,), dtype=np.float32)
        obs_dim = 13 + (2 * self.num_joints) + 2 
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)


        self.ambient_temp = 21.1
        self.temperatures = np.full(self.num_joints, self.ambient_temp)
        self.current_action = np.zeros(self.num_joints)
        
        self.cooling_factor = 0.9927 
        self.heating_factor = 0.425
        self.REAL_MAX_TORQUE = 0.00328 * 1e-3 
        self.FORCE_SCALING = 5000.0 

        self.wave_freq = 0.03  
        self.wave_lag = 1.5    

        self.current_step = 0
        self.max_steps = 2000

    def step(self, action):
        self.current_step += 1
        self.current_action = action
        t = self.current_step
        

        base_signal = np.array([math.sin(t * self.wave_freq - i * self.wave_lag) for i in range(self.num_joints)])
        

        # action(0~1) -> (-1~1)로 변환 후, 0.5배 축소하여 '보정' 용도로만 사용
        ai_correction = (action - 0.5) * 2.0 

        # 이렇게 하면 AI가 조금만 실수해도 기본 파동 덕분에 앞으로 잘 갑니다.
        # 즉, "가만히 있으면 중간은 간다"는 걸 알기에 엔트로피가 낮아집니다.
        final_command = base_signal + (ai_correction * 0.5)
        
        for i, joint_idx in enumerate(self.joints):
            target_angle = final_command[i] * 1.5 
            torque = self.REAL_MAX_TORQUE * self.FORCE_SCALING
            
            p.setJointMotorControl2(
                self.robot_id, joint_idx, p.POSITION_CONTROL,
                targetPosition=target_angle, force=torque
            )

        p.stepSimulation()

        
        base_vel, _ = p.getBaseVelocity(self.robot_id)
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        roll, pitch, yaw = p.getEulerFromQuaternion(base_orn)

        
        reward_forward = base_vel[0] * 200.0 
        
        
        reward_energy = -np.mean(np.abs(action - 0.5)) * 0.01
        
        reward_stability = -(abs(roll) + abs(pitch)) * 0.1

        reward_lazy = -1.0 if base_vel[0] < 0.005 else 0.0

        reward = reward_forward + reward_energy + reward_stability + reward_lazy

        terminated = False
        if base_pos[2] > 0.5 or abs(roll) > 1.5 or abs(pitch) > 1.5: 
            terminated = True
            reward -= 50.0

        truncated = (self.current_step >= self.max_steps)
        
        return self._get_obs(), reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        p.resetBasePositionAndOrientation(self.robot_id, self.start_pos, self.start_orn)
        self.temperatures = np.full(self.num_joints, self.ambient_temp)
        self.current_action = np.zeros(self.num_joints)
        
        for i in self.joints:
            p.resetJointState(self.robot_id, i, targetValue=0.0)
            
        return self._get_obs(), {}

    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id)
        joint_states = p.getJointStates(self.robot_id, self.joints)
        j_angles = [s[0] for s in joint_states]
        j_vels = [s[1] for s in joint_states]
        
        freq = self.wave_freq 
        phase = [math.sin(self.current_step * freq), math.cos(self.current_step * freq)]

        obs = np.concatenate([
            pos, orn, lin_vel, ang_vel, 
            j_angles, j_vels, 
            phase 
        ])
        return obs.astype(np.float32)

    def close(self):
        p.disconnect()
