import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from gan_engine import AliceNet, EveNet, set_seed

# =================================================================
# 1. 实验基础配置
# =================================================================
MSG_LEN = 16
KEY_LEN = 16
CHECKPOINT_PATH = "best_checkpoint.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义采样规模梯度 (2^7 到 2^14)
SAMPLE_SIZES = [128, 512, 1024, 4096, 8192, 16384]

def load_alice():
    alice = AliceNet(MSG_LEN, KEY_LEN).to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        state_dict = checkpoint['alice'] if 'alice' in checkpoint else checkpoint
        alice.load_state_dict(state_dict)
    alice.eval()
    return alice

def train_attacker(alice, train_size, epochs=2000):
    """模拟攻击者 Eve 在已知 train_size 组数据下的破译过程"""
    set_seed(42)
    eve = EveNet(MSG_LEN).to(DEVICE)
    optimizer = optim.Adam(eve.parameters(), lr=0.001)
    criterion = nn.L1Loss()
    
    # 构造已知明密文对 (训练集)
    with torch.no_grad():
        p_train = torch.randint(0, 2, (train_size, MSG_LEN)).float().to(DEVICE) * 2 - 1
        k_train = torch.randn(train_size, KEY_LEN).to(DEVICE)
        c_train = alice(p_train, k_train)
    
    # 训练 Eve 尝试拟合 P = f(C)
    for epoch in range(epochs):
        optimizer.zero_grad()
        p_guess = eve(c_train)
        loss = criterion(p_guess, p_train)
        loss.backward()
        optimizer.step()
    
    # 在独立测试集上评估破译准确率
    with torch.no_grad():
        p_test = torch.randint(0, 2, (1000, MSG_LEN)).float().to(DEVICE) * 2 - 1
        k_test = torch.randn(1000, KEY_LEN).to(DEVICE)
        c_test = alice(p_test, k_test)
        
        p_recovered = (eve(c_test) > 0).float()
        p_target = (p_test > 0).float()
        # 【修复点】：增加 .float() 转换
        acc = (p_recovered == p_target).float().mean().item()
    
    return acc

# =================================================================
# 3. 执行实验
# =================================================================
if __name__ == "__main__":
    print("[*] 正在启动实验 4: 抗已知明文攻击 (KPA) 破译压力测试...")
    alice = load_alice()
    
    results = []
    for size in SAMPLE_SIZES:
        print(f"  > 正在测试 Eve 的破译上限 | 已知样本数: {size}...")
        final_acc = train_attacker(alice, size)
        results.append(final_acc)
        print(f"    [!] 最终破译准确率: {final_acc:.2%}")

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(SAMPLE_SIZES, results, 'r-o', linewidth=2, markersize=8, label="Eve's Recovery Acc")
    plt.axhline(y=0.5, color='gray', linestyle='--', label='Baseline (Random Guess)')
    plt.axhline(y=1.0, color='blue', linestyle=':', label='Total Crack (100%)')
    
    plt.xscale('log')
    plt.title("Experiment 4: KPA Resistance Learning Curve", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Known Plaintext-Ciphertext Pairs (Log Scale)", fontsize=12)
    plt.ylabel("Eve's Bit-wise Recovery Accuracy", fontsize=12)
    plt.ylim(0.4, 1.05)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig("exp4_kpa_resistance.png", dpi=300)
    print("\n[√] 实验 4 完成。破译抗性曲线已保存: exp4_kpa_resistance.png")