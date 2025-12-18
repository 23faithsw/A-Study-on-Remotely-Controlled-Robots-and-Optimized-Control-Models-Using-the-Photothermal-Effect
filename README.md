# A Study on Remotely Controlled Robots and Optimized Control Models Using the Photothermal Effect
[열시스템해석] 팀 프로젝트(논문 작성)

-----

## Research Topic: A Study on Remotely Controlled Robots and Optimized Control Models Using the Photothermal Effect
## Research Methods

  * Building a simulation environment using **PyBullet**.
  * Implementing **Reinforcement Learning** using the **SAC (Soft Actor-Critic)** deep learning model.

## Key Results

  * **<img width="563" height="362" alt="rew_mean" src="https://github.com/user-attachments/assets/333efae9-a4a0-4f69-a14d-4703a7f6a8d0" />, <img width="843" height="363" alt="enf" src="https://github.com/user-attachments/assets/c32c215f-efb5-436c-b20f-e265a68ee316" />. <img width="640" height="364" alt="enf_coef_loss" src="https://github.com/user-attachments/assets/b9fb8526-8a6b-44e1-bb4b-2121294d7207" />**

      * *Training results of the SAC model (TensorBoard logs)*

    
  * **<img width="1858" height="1053" alt="발표용 그래프" src="https://github.com/user-attachments/assets/4d6f47b8-83ef-4579-8db9-e9277e630e12" />**

      * *Performance analysis of the optimized model: Joint actuation heatmap and 2D trajectory*

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install gymnasium stable-baselines3 pybullet shimmy numpy matplotlib\
```

### 2. Train the Agent

```bash
python train_final.py
```
Training logs will be saved in the final_logs directory.

### 3. Verify & Visualize Results

```bash
python verify_emergence.py
```
This script loads the trained model and generates performance analysis graphs.
