# BIoMT + 动态区块链配置

本工程实现（' 通道选择 + VRF + 强化学习对被选通道配置 m/n + αβγ 状态驱动 + TxPool 不淘汰），支持：

- **训练集：Set-A + Outcomes-a**
- **测试集：Set-B**

## 目录结构

```
physionet2012/
  set-a/
    132539.txt
    ...
  set-b/
    140501.txt
    ...
  Outcomes-a.txt

## 运行

### 训练（用 Set-A）
```bash
pip install -r requirements.txt
python -m rl.train_rl_ppo --mode train --set_a_dir <path>/set-a --outcomes_a <path>/Outcomes-a.txt
```

### 评估（用 Set-B，只 rollout 不更新）
```bash
python -m rl.train_rl_ppo --mode eval --set_b_dir <path>/set-b
```



## 输出
- 训练/评估日志：`logs/train_log.csv` 或 `logs/eval_log.csv`
- 画图：`python -m logs.plot_metrics`（读最近的 CSV）
