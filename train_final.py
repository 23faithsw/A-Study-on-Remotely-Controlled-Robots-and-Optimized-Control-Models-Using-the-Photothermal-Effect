import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import shutil
from LCP_CaterpillarEnv_Final import LCP_CaterpillarEnv


# 2. 환경 및 모델 설정
env = LCP_CaterpillarEnv(render=False) # 학습 속도를 위해 렌더링 끔

print("--- 🚀 S자 주행 학습 시작 (Residual RL) ---")
checkpoint_callback = CheckpointCallback(
save_freq=10000,
save_path="./models",
name_prefix="sac_lce"
)

# 3. 모델 생성 (49차원 입력 자동 인식)

model = SAC(
"MlpPolicy",
env,
verbose=1,
tensorboard_log="./final_logs",
learning_rate=3e-4,
batch_size=256,
ent_coef='auto'
)

# 4. 학습 실행
model.learn(total_timesteps=50000, log_interval=1, callback=checkpoint_callback)

model.save("sac_lce_final_model")

print("--- ✅ 학습 완료! 이제 verify 코드를 돌려보세요. ---")