# A Study on Remotely Controlled Robots and Optimized Control Models Using the Photothermal Effect
[열시스템해석] 팀 프로젝트(논문 작성)

-----

## Research Topic: Control Optimization of a Remotely Operated Robot using the Photothermal Effect

## Research Methods

  * Building a simulation environment using **PyBullet**.
  * Implementing **Reinforcement Learning** using the **SAC (Soft Actor-Critic)** deep learning model.

## Key Results

  * **(Attach TensorBoard training graph screenshot here)**
      * *Description: The training curve (e.g., `rollout/ep_rew_mean`) shows the learning progress over 500k+ steps.*
  * **(Attach final optimized light pattern heatmap screenshot here)**
      * *Description: The final heatmap visualizes the optimized light actuation pattern learned by the SAC agent.*

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install gymnasium stable-baselines3 pybullet shimmy numpy matplotlib

### 2. Train the Agent
Bash

python train_final.py
Training logs will be saved in the final_logs directory.

### 3. Verify & Visualize Results
Bash

python verify_emergence.py
This script loads the trained model and generates performance analysis graphs.
    ```bash
    # To visualize the trained agent's behavior and plot the results
    python test_and_plot.py
    ```
