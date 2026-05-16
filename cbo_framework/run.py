"""
入口脚本 - 运行因果贝叶斯优化
Usage: python run.py
"""

import sys
import os

# ============================================================
# 第一步：设置Atlatl路径
# 修改下面这行为你的实际路径
# ============================================================
ATLATL_SERVER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "atlatl-public-master", "server"
)
ATLATL_SERVER = os.path.abspath(ATLATL_SERVER)

# 将Atlatl server目录加入Python搜索路径
# 这样 import map, import unit, import combat 等才能工作
if ATLATL_SERVER not in sys.path:
    sys.path.insert(0, ATLATL_SERVER)

# 验证路径
if not os.path.exists(os.path.join(ATLATL_SERVER, "game.py")):
    print(f"错误：找不到Atlatl代码")
    print(f"检查路径: {ATLATL_SERVER}")
    print(f"确保 atlatl-public-master/ 和 cbo_framework/ 在同一目录下")
    print(f"目录结构应该是:")
    print(f"  your_project/")
    print(f"    atlatl-public-master/")
    print(f"      server/")
    print(f"        game.py  <-- 需要找到这个文件")
    print(f"    cbo_framework/")
    print(f"      run.py     <-- 你在这里")
    sys.exit(1)

print(f"Atlatl路径: {ATLATL_SERVER}")

# 更新config中的路径
import config
config.ATLATL_SERVER_PATH = ATLATL_SERVER

# ============================================================
# 第二步：运行CBO
# ============================================================
from cbo_optimizer import CBOOptimizer

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="因果贝叶斯优化 - Atlatl对抗鲁棒配置")
    parser.add_argument("--n_initial", type=int, default=10,
                        help="初始随机采样数 (default: 10)")
    parser.add_argument("--n_iter", type=int, default=15,
                        help="CBO迭代次数 (default: 15)")
    parser.add_argument("--n_d", type=int, default=30,
                        help="每轮d候选数 (default: 30)")
    parser.add_argument("--n_u", type=int, default=50,
                        help="每轮u候选数 (default: 50)")
    parser.add_argument("--n_seeds", type=int, default=2,
                        help="每次评估重复次数 (default: 2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (default: 42)")
    parser.add_argument("--output", type=str, default="results.json",
                        help="输出文件 (default: results.json)")
    args = parser.parse_args()

    print(f"\n配置:")
    print(f"  初始采样: {args.n_initial}")
    print(f"  CBO迭代: {args.n_iter}")
    print(f"  d候选数: {args.n_d}")
    print(f"  u候选数: {args.n_u}")
    print(f"  评估种子数: {args.n_seeds}")
    print(f"  随机种子: {args.seed}")

    optimizer = CBOOptimizer(
        n_initial=args.n_initial,
        n_iterations=args.n_iter,
        n_d_candidates=args.n_d,
        n_u_candidates=args.n_u,
        n_eval_seeds=args.n_seeds,
        seed=args.seed,
    )

    results = optimizer.optimize()
    optimizer.save_results(args.output)
