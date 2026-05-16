# 基于因果贝叶斯优化的Atlatl对抗鲁棒配置搜索

## 项目结构

```
your_project/
├── atlatl-public-master/       # Atlatl源码（解压你的zip）
│   ├── server/
│   │   ├── game.py
│   │   ├── combat.py
│   │   ├── mobility.py
│   │   ├── map.py
│   │   ├── unit.py
│   │   ├── status.py
│   │   ├── ai/
│   │   │   ├── pass_agg.py
│   │   │   ├── passive.py
│   │   │   ├── shootback.py
│   │   │   ├── random_actor.py
│   │   │   └── ...
│   │   └── ...
│   └── browser/
├── cbo_framework/              # 本项目代码
│   ├── config.py               # 变量定义 + 因果边 + 路径配置
│   ├── causal_graph.py         # 因果DAG数据结构
│   ├── causal_gp.py            # 因果结构化高斯过程
│   ├── acquisition.py          # minimax采集函数
│   ├── atlatl_evaluator.py     # Atlatl同步评估器
│   ├── cbo_optimizer.py        # 主优化循环
│   └── run.py                  # 入口脚本
└── README.md
```

## 环境要求

- Python 3.9+
- numpy
- scipy
- websockets（Atlatl的AI模块依赖）

## 安装步骤

### 1. 解压Atlatl

```bash
unzip atlatl-public-master.zip
```

### 2. 安装Python依赖

```bash
pip install numpy scipy websockets
```

### 3. 配置路径

打开 `cbo_framework/config.py`，修改第6行：

```python
ATLATL_SERVER_PATH = "/你的路径/atlatl-public-master/server"
```

改成你实际解压Atlatl的路径。例如：
- Windows: `ATLATL_SERVER_PATH = "C:/Users/xxx/atlatl-public-master/server"`
- Linux/Mac: `ATLATL_SERVER_PATH = "/home/xxx/atlatl-public-master/server"`

### 4. 运行

```bash
cd cbo_framework
python run.py
```

## 运行参数

在 `run.py` 中可以调整：

```python
optimizer = CBOOptimizer(
    n_initial=20,       # 初始随机采样数（建议>=15）
    n_iterations=30,    # CBO迭代次数（越多越精确）
    n_d_candidates=50,  # 每轮d候选数
    n_u_candidates=100, # 每轮u候选数
    n_eval_seeds=3,     # 每次评估重复次数（降噪）
    seed=42,            # 随机种子
)
```

快速测试用小参数：`n_initial=8, n_iterations=10, n_eval_seeds=2`
正式实验用大参数：`n_initial=30, n_iterations=50, n_eval_seeds=5`

## 输出解读

运行结束后会打印：

1. **最优鲁棒配置 d***：在最坏环境下表现最好的蓝方配置
2. **因果分解**：得分中每条因果路径的贡献比例
3. **脆弱性分析**：d*相对最差情况在每条路径上的优势
4. **因果路径**：从输入变量到得分的完整因果链

结果同时保存到 `results.json`。
