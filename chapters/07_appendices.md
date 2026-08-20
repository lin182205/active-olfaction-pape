# 附录A 当前代码关键参数汇总

表A-1 论文初稿对应的主要默认参数

| **模块**    | **参数**               | **当前值**     |
|-------------|------------------------|----------------|
| 移动环境    | world_half             | 1.0 m          |
| 移动环境    | max_steps              | 400            |
| 移动环境    | goal_radius            | 0.10 m         |
| 底盘        | forward_speed          | 0.15 m/s       |
| 底盘        | turn_rate              | 1.5 rad/s      |
| 环境        | dt                     | 0.2 s          |
| 触须        | length                 | 0.3 m          |
| 触须        | sector_count           | 10/侧          |
| 触须        | servo_60deg_time       | 0.12 s         |
| 传感器      | sensor_model           | asymmetric     |
| 传感器      | response/recovery tau  | 0.6 / 3.0 s    |
| 传感器      | noise_std              | 0.006          |
| 观测        | single-frame dim       | 16             |
| 观测        | history length         | 20             |
| 时序        | default encoder        | GRU            |
| GRU         | hidden/layers/features | 64 / 1 / 64    |
| Transformer | d_model/heads/layers   | 64 / 4 / 2     |
| Transformer | FF/dropout/features    | 128 / 0.1 / 64 |
| PPO         | learning rate          | 1e−4           |
| PPO         | n_envs × n_steps       | 8 × 512        |
| PPO         | batch / epochs         | 256 / 5        |
| PPO         | gamma / lambda         | 0.99 / 0.95    |
| PPO         | clip range / ent coef  | 0.2 / 0.003    |
| PPO         | target KL              | 0.02           |

# 附录B 论文内容与代码模块对应关系

表B-1 方法描述与仓库代码对应关系

| **论文内容**                 | **主要代码文件**                                              |
|------------------------------|---------------------------------------------------------------|
| 移动机器人联合搜索环境       | dual_whisker_rl/envs/mobile_whisker_env.py                    |
| 动态 puff 羽流与基础触须环境 | dual_whisker_rl/envs/whisker_only_env.py                      |
| 底盘离散动力学               | dual_whisker_rl/envs/robot_model.py                           |
| 双触须扇区与采样点几何       | dual_whisker_rl/envs/whisker_model.py                         |
| 一阶/非对称气体传感器        | dual_whisker_rl/envs/sensor_model.py                          |
| 硬件一致 12 维观测           | dual_whisker_rl/observation/whisker_observation_builder.py    |
| MQ-3 在线预处理              | dual_whisker_rl/hardware/sensor_preprocess.py                 |
| GRU 历史编码                 | dual_whisker_rl/agents/gru_history.py                         |
| Transformer 历史编码         | dual_whisker_rl/agents/transformer_history.py                 |
| 移动触须 PPO 训练            | scripts/train_mobile_whisker_ppo.py                           |
| 统一评估指标                 | dual_whisker_rl/evaluation.py                                 |
| STM32 接口配置               | configs/hardware.yaml + hardware/firmware/stm32_dual_whisker/ |
