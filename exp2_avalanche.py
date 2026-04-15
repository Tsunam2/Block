import torch
import numpy as np
import matplotlib.pyplot as plt
from Crypto.Cipher import AES
import os
from gan_engine import AliceNet, set_seed

# =================================================================
# 1. 实验基础配置
# =================================================================
MSG_LEN = 16
KEY_LEN = 16
CHECKPOINT_PATH = "best_checkpoint.pth"
SAMPLE_COUNT = 10000  # 每个比特位采样 1 万次以获得稳定的期望值
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    alice = AliceNet(MSG_LEN, KEY_LEN).to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        state_dict = checkpoint['alice'] if 'alice' in checkpoint else checkpoint
        alice.load_state_dict(state_dict)
    alice.eval()
    return alice

# =================================================================
# 2. 实验 2a: 明文严格雪崩准则 (Plaintext SAC)
# =================================================================
def run_plaintext_sac(alice):
    print("[*] 正在执行实验 2a: 明文严格雪崩准则测试...")
    set_seed(42)
    
    nce_bfr = []
    aes_bfr = []

    # AES 基础设置
    aes_key = os.urandom(16)
    cipher = AES.new(aes_key, AES.MODE_ECB)

    for bit_idx in range(MSG_LEN):
        # --- NCE 测试 ---
        m1 = torch.randint(0, 2, (SAMPLE_COUNT, MSG_LEN)).float().to(DEVICE) * 2 - 1
        m2 = m1.clone()
        # 翻转指定比特: -1 -> 1, 1 -> -1
        m2[:, bit_idx] = -m2[:, bit_idx]
        
        k = torch.randn(SAMPLE_COUNT, KEY_LEN).to(DEVICE)
        
        with torch.no_grad():
            # 这里的阈值判定即模拟比特层面的翻转情况
            c1 = (alice(m1, k) > 0).int()
            c2 = (alice(m2, k) > 0).int()
            # 计算比特反转率 (BFR)
            diff = (c1 != c2).float().mean().item()
            nce_bfr.append(diff)

        # --- AES 测试 ---
        # 生成随机 16 字节明文作为对比基准
        p1_bytes = os.urandom(SAMPLE_COUNT * 16)
        c1_aes = cipher.encrypt(p1_bytes)
        
        # 翻转指定比特位
        p1_arr = np.frombuffer(p1_bytes, dtype=np.uint8).copy()
        byte_pos = bit_idx // 8
        bit_pos = bit_idx % 8
        p1_arr.reshape(-1, 16)[:, byte_pos] ^= (1 << bit_pos)
        
        c2_aes = cipher.encrypt(p1_arr.tobytes())
        
        # 计算 AES 的比特反转率
        c1_bits = np.unpackbits(np.frombuffer(c1_aes, dtype=np.uint8))
        c2_bits = np.unpackbits(np.frombuffer(c2_aes, dtype=np.uint8))
        diff_aes = (c1_bits != c2_bits).mean()
        aes_bfr.append(diff_aes)

    # 绘图 2a
    plt.figure(figsize=(10, 6))
    plt.plot(range(MSG_LEN), nce_bfr, 'o-', label='Proposed NCE', color='#1f77b4', linewidth=2)
    plt.plot(range(MSG_LEN), aes_bfr, 's--', label='AES-128', color='#d62728', alpha=0.7)
    plt.axhline(y=0.5, color='black', linestyle=':', label='Ideal SAC (0.5)', linewidth=1.5)
    
    plt.title("Experiment 2a: Plaintext Strict Avalanche Criterion (SAC)", fontsize=14, fontweight='bold')
    plt.xlabel("Index of Flipped Bit in Message", fontsize=12)
    plt.ylabel("Mean Bit Flip Ratio (BFR)", fontsize=12)
    plt.ylim(0, 1.0)
    plt.xticks(range(MSG_LEN))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("exp2_a_msg_sensitivity.png", dpi=300)
    print("[+] 已保存: exp2_a_msg_sensitivity.png")

# =================================================================
# 3. 实验 2b: 密钥敏感度测试 (Key Sensitivity)
# =================================================================
def run_key_sensitivity(alice):
    print("[*] 正在执行实验 2b: 密钥敏感度测试 (噪声注入)...")
    set_seed(42)
    
    # 噪声强度等级 (对数刻度: 从极小噪声到极大噪声)
    noise_levels = np.logspace(-6, 0, 20)
    uaci_scores = []

    m = torch.randint(0, 2, (SAMPLE_COUNT, MSG_LEN)).float().to(DEVICE) * 2 - 1
    k_base = torch.randn(SAMPLE_COUNT, KEY_LEN).to(DEVICE)
    
    with torch.no_grad():
        c_base = alice(m, k_base)
        
        for eps in noise_levels:
            # 注入噪声: k_new = k_base + GaussianNoise(0, eps)
            k_perturbed = k_base + torch.randn_like(k_base) * eps
            c_perturbed = alice(m, k_perturbed)
            
            # 计算密文漂移强度 (Mean L1 Distance)
            # 在神经网络加密中，这比 BFR 更能体现非线性变化的连续敏感性
            diff_intensity = torch.mean(torch.abs(c_base - c_perturbed)).item()
            uaci_scores.append(diff_intensity)

    # 绘图 2b
    plt.figure(figsize=(10, 6))
    plt.semilogx(noise_levels, uaci_scores, 'D-', color='#27ae60', linewidth=2, markersize=6)
    plt.fill_between(noise_levels, uaci_scores, color='#27ae60', alpha=0.1)
    
    plt.title("Experiment 2b: Key Sensitivity Analysis (Confusion)", fontsize=14, fontweight='bold')
    plt.xlabel("Key Perturbation Strength (Log Scale: Sigma)", fontsize=12)
    plt.ylabel("Ciphertext Change Intensity (Mean L1)", fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    plt.savefig("exp2_b_key_sensitivity.png", dpi=300)
    print("[+] 已保存: exp2_b_key_sensitivity.png")

if __name__ == "__main__":
    model = load_model()
    run_plaintext_sac(model)
    run_key_sensitivity(model)
    print("\n[√] 实验 2 完成。两张雪崩效应评估图表已生成。")