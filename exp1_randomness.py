import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Crypto.Cipher import AES
import os
from scipy.special import erfc
from gan_engine import AliceNet, set_seed

# =================================================================
# 1. 实验基础配置
# =================================================================
MSG_LEN = 16
KEY_LEN = 16
CHECKPOINT_PATH = "best_checkpoint.pth"
BIT_COUNT = 1000000  # 采样 100 万比特用于统计
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_bitstreams():
    """获取实验数据：修复了 AES 对齐问题，并确保所有返回值为 NumPy 数组"""
    set_seed(42)
    
    # --- A. 神经加密引擎 (NCE) ---
    alice = AliceNet(MSG_LEN, KEY_LEN).to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        # 针对你的 checkpoint 结构进行适配
        state_dict = checkpoint['alice'] if 'alice' in checkpoint else checkpoint
        alice.load_state_dict(state_dict)
    alice.eval()

    num_batches = BIT_COUNT // MSG_LEN + 1
    nce_bits_list = []
    nce_cont_list = [] 

    print("[*] 正在通过神经网络采样 100 万比特数据...")
    with torch.no_grad():
        # 分批生成，每批 500 条消息
        for _ in range((num_batches // 500) + 1):
            m = torch.randint(0, 2, (500, MSG_LEN)).float().to(DEVICE) * 2 - 1
            k = torch.randn(500, KEY_LEN).to(DEVICE)
            c = alice(m, k)
            c_np = c.cpu().numpy().flatten()
            nce_cont_list.extend(c_np)
            nce_bits_list.extend((c_np > 0).astype(int))
    
    # --- B. AES-128 逻辑 (修复对齐问题) ---
    print("[*] 正在通过 AES-128 采样对比数据...")
    aes_key = os.urandom(16)
    cipher = AES.new(aes_key, AES.MODE_ECB)
    
    # 计算 16 字节对齐长度
    needed_bytes = BIT_COUNT // 8
    aligned_bytes = ((needed_bytes + 15) // 16) * 16
    
    aes_raw_bytes = os.urandom(aligned_bytes)
    aes_cipher_bytes = cipher.encrypt(aes_raw_bytes)
    
    aes_uint8 = np.frombuffer(aes_cipher_bytes, dtype=np.uint8)
    # 正确映射到 [-1, 1]
    aes_continuous = (aes_uint8.astype(float) / 127.5) - 1.0 
    aes_bits = np.unpackbits(aes_uint8)

    # --- C. 理想随机 ---
    rand_bits = np.random.randint(0, 2, BIT_COUNT)

    # 【重要修复】：将所有 list 转换为 numpy 数组，并截断到统一长度
    return (
        np.array(nce_bits_list[:BIT_COUNT]), 
        np.array(nce_cont_list[:BIT_COUNT]), 
        np.array(aes_bits[:BIT_COUNT]), 
        np.array(aes_continuous[:BIT_COUNT]), 
        np.array(rand_bits[:BIT_COUNT])
    )

# =================================================================
# 2. 独立绘图函数
# =================================================================

def plot_a_pdf(nce_cont, aes_cont):
    """图 A: 概率密度分布 (对应信息熵)"""
    plt.figure(figsize=(10, 6))
    # 增加平滑度以便观察
    sns.kdeplot(nce_cont, color='#1f77b4', fill=True, label='Proposed NCE', bw_adjust=0.5, linewidth=2)
    sns.kdeplot(aes_cont, color='#d62728', fill=False, label='AES-128', linestyle='--', linewidth=2)
    
    plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.6, label='Ideal Uniform Target')
    
    plt.title("Ciphertext Probability Density Distribution", fontsize=14, fontweight='bold')
    plt.xlabel("Value Space (Normalized [-1, 1])", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.ylim(0, 1.2) # 适当留白
    plt.legend(loc='upper center')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig("exp1_a_distribution.png", dpi=300)
    print("[+] 已保存: exp1_a_distribution.png")

def plot_b_frequency(nce_bits, aes_bits, rand_bits):
    """图 B: 频率测试 (0/1 比例)"""
    labels = ['NCE', 'AES-128', 'True Random']
    zeros = [np.mean(nce_bits == 0), np.mean(aes_bits == 0), np.mean(rand_bits == 0)]
    ones = [np.mean(nce_bits == 1), np.mean(aes_bits == 1), np.mean(rand_bits == 1)]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, zeros, width, label='Bit 0', color='#3498db', edgecolor='black', alpha=0.8)
    ax.bar(x + width/2, ones, width, label='Bit 1', color='#e74c3c', edgecolor='black', alpha=0.8)

    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1.5, label='Ideal Ratio (0.5)')
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title('NIST Frequency Test: Monobit Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.7)
    ax.legend()
    plt.tight_layout()
    plt.savefig("exp1_b_frequency.png", dpi=300)
    print("[+] 已保存: exp1_b_frequency.png")

def plot_c_block_frequency(nce_bits, aes_bits):
    """图 C: 块内频率波动 (箱线图)"""
    block_size = 128
    num_blocks = 1000 
    
    def get_block_probs(bits):
        # 确保输入是 numpy 数组
        bits_arr = np.array(bits)
        reshaped = bits_arr[:num_blocks * block_size].reshape(num_blocks, block_size)
        return np.mean(reshaped, axis=1)

    data = [get_block_probs(nce_bits), get_block_probs(aes_bits)]
    
    plt.figure(figsize=(10, 6))
    # 适配 Matplotlib 3.9+ 的 tick_labels 参数，同时兼容旧版本
    try:
        box = plt.boxplot(data, patch_artist=True, tick_labels=['Proposed NCE', 'AES-128'])
    except:
        box = plt.boxplot(data, patch_artist=True, labels=['Proposed NCE', 'AES-128'])
    
    colors = ['#3498db', '#e74c3c']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.8)
    plt.title("NIST Block Frequency: Local Density Stability", fontsize=14, fontweight='bold')
    plt.ylabel("Proportion of '1' per 128-bit Block")
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig("exp1_c_block_frequency.png", dpi=300)
    print("[+] 已保存: exp1_c_block_frequency.png")

def plot_d_runs(nce_bits, aes_bits):
    """图 D: 游程长度分布 (对数拟合)"""
    def get_run_counts(bits, max_run=8):
        counts = []
        bits_arr = np.array(bits)
        for i in range(1, max_run + 1):
            match_count = 0
            curr_run = 0
            for b in bits_arr:
                if b == 1: curr_run += 1
                else:
                    if curr_run == i: match_count += 1
                    curr_run = 0
            counts.append(match_count)
        # 归一化
        res = np.array(counts)
        return res / np.sum(res)

    runs_x = np.arange(1, 9)
    nce_runs = get_run_counts(nce_bits)
    aes_runs = get_run_counts(aes_bits)
    
    # 理论理想曲线: 1/2^n
    ideal_runs = 1 / (2**runs_x)
    ideal_runs /= np.sum(ideal_runs)

    plt.figure(figsize=(10, 6))
    plt.plot(runs_x, nce_runs, 'o-', label='Proposed NCE', color='#1f77b4', linewidth=2)
    plt.plot(runs_x, aes_runs, 's--', label='AES-128', color='#d62728', alpha=0.7)
    plt.plot(runs_x, ideal_runs, 'k:', label='Theoretical Ideal (1/2^n)', linewidth=2)

    plt.yscale('log')
    plt.title("NIST Runs Test: Run-length Decay Distribution", fontsize=14, fontweight='bold')
    plt.xlabel("Run Length (n)", fontsize=12)
    plt.ylabel("Log Frequency", fontsize=12)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig("exp1_d_runs.png", dpi=300)
    print("[+] 已保存: exp1_d_runs.png")

# =================================================================
# 3. 主程序入口
# =================================================================
if __name__ == "__main__":
    print("[*] 正在启动安全性评估 - 实验 1: 随机性与比特分布分析...")
    
    try:
        nce_b, nce_c, aes_b, aes_c, rand_b = get_bitstreams()
        
        plot_a_pdf(nce_c, aes_c)
        plot_b_frequency(nce_b, aes_b, rand_b)
        plot_c_block_frequency(nce_b, aes_b)
        plot_d_runs(nce_b, aes_b)
        
        print("\n[√] 实验 1 成功完成！所有学术图表已更新。")
    except Exception as e:
        print(f"\n[!] 实验运行失败: {e}")
        import traceback
        traceback.print_exc()