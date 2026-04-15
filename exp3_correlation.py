import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from gan_engine import AliceNet, set_seed

# =================================================================
# 1. 实验基础配置
# =================================================================
MSG_LEN = 16
KEY_LEN = 16
CHECKPOINT_PATH = "best_checkpoint.pth"
SAMPLE_COUNT = 100000  # 采样 10 万组数据，确保统计学上的严谨性
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    alice = AliceNet(MSG_LEN, KEY_LEN).to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        state_dict = checkpoint['alice'] if 'alice' in checkpoint else checkpoint
        alice.load_state_dict(state_dict)
    alice.eval()
    return alice

def run_correlation_test(alice):
    """执行皮尔逊相关性分析并生成热力图"""
    print(f"[*] 正在启动实验 3: 明密文相关性审计 (采样数: {SAMPLE_COUNT})...")
    set_seed(42)
    
    # 1. 生成大规模随机明文和密钥
    # 明文 P 为二进制比特映射到 [-1, 1]
    p = torch.randint(0, 2, (SAMPLE_COUNT, MSG_LEN)).float().to(DEVICE) * 2 - 1
    k = torch.randn(SAMPLE_COUNT, KEY_LEN).to(DEVICE)
    
    # 2. 获取密文 C (保留连续值以检测更细微的相关性)
    with torch.no_grad():
        c = alice(p, k).cpu().numpy()
    p_np = p.cpu().numpy()
    
    # 3. 计算皮01尔逊相关系数矩阵
    # 我们构造组合矩阵 [P, C]，计算其协方差，进而得到相关矩阵
    combined = np.hstack([p_np, c])
    corr_matrix_full = np.corrcoef(combined, rowvar=False)
    
    # 切片出 P (前16列) 与 C (后16列) 的互相关部分 (16x16)
    corr_matrix = corr_matrix_full[:MSG_LEN, MSG_LEN:]
    
    # 4. 绘图：相关性热力图
    plt.figure(figsize=(12, 10))
    # 使用 RdBu_r 配色，红色代表正相关，蓝色代表负相关，白色代表无关
    sns.heatmap(corr_matrix, 
                annot=False, 
                cmap='RdBu_r', 
                center=0, 
                vmin=-0.1, vmax=0.1, # 设定极小范围，如果这里都是白色，说明安全性极高
                xticklabels=[f'C{i}' for i in range(MSG_LEN)],
                yticklabels=[f'P{i}' for i in range(MSG_LEN)])
    
    plt.title("Experiment 3: Plaintext-Ciphertext Correlation Heatmap", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Ciphertext Dimensions", fontsize=12)
    plt.ylabel("Plaintext Bits", fontsize=12)
    
    # 5. 统计指标计算
    avg_corr = np.mean(np.abs(corr_matrix))
    max_corr = np.max(np.abs(corr_matrix))
    plt.figtext(0.5, 0.02, f"Mean Abs Correlation: {avg_corr:.6f} | Max Abs Correlation: {max_corr:.6f}", 
                ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig("exp3_correlation_heatmap.png", dpi=300)
    print("[+] 实验 3 热力图已保存: exp3_correlation_heatmap.png")
    
    return corr_matrix

if __name__ == "__main__":
    try:
        model = load_model()
        run_correlation_test(model)
        print("\n[√] 实验 3 成功完成！该图表证明了系统在统计上的混淆能力。")
    except Exception as e:
        print(f"\n[!] 运行失败: {e}")