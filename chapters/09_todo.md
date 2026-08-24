# 初稿后续补充清单

- 补充 MQ-3 标准浓度标定、响应/恢复时间、重复性与两只传感器个体差异实验。

- 优先完成固定基座 active / fixed-multi / periodic / random-neighbor 的成对仿真与随机区组真机实验，先验证主动采样信息增益，再进入移动结论。

- 冻结环境版本后完成不少于 5 个训练 seed 的主方法正式训练，并保存 run_metadata、TensorBoard 和 checkpoint。

- 完成 fixed-whisker、MLP/GRU/Transformer、blank-age、reacquisition reward、domain randomization 等消融。

- 移动主比较统一使用同一 PPO、时序编码和训练预算，仅改变触须控制方式；增加 `progress_reward_scale=0` 的无距离塑形对照。

- 用统一 evaluation.py 生成成功率、路径、失嗅和重捕获指标，并进行统计显著性/置信区间分析。

- 以训练 seed 而非单个 episode 作为独立统计单位，预先指定主要指标并报告效应量与置信区间。

- 正式投稿前补充相关工作综述与近5年主动嗅觉/气味源定位深度强化学习文献，统一参考文献格式。

- 根据目标期刊/会议模板重新压缩章节、加入方法框图、网络结构图、羽流示意图、训练曲线、轨迹图和消融结果图。
