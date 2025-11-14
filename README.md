# Control-Optimization-of-a-Remotely-Operated-Robot-using-the-Photothermal-Effect
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

## How to Run

1.  **Create and activate the conda environment:**

    ```bash
    conda create -n lcp_env python=3.10
    conda activate lcp_env
    ```

2.  **Install dependencies:**

    ```bash
    pip install ... 
    # (Add your full pip install commands here)
    ```

3.  **Train the model:**

    ```bash
    # To train the agent
    python train.py
    ```

4.  **Test and visualize the results:**

    ```bash
    # To visualize the trained agent's behavior and plot the results
    python test_and_plot.py
    ```
