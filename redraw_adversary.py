import torch
import numpy as np
import matplotlib.pyplot as plt

# =================================================================
# 核心精修绘图脚本 (直接加载本地 .pt 数据，无需重新训练)
# =================================================================

def redraw_final_plots():
    # 1. 尝试加载原始实验数据
    try:
        results_knowledge = torch.load("raw_data_knowledge.pt", weights_only=False)
        results_capacity = torch.load("raw_data_capacity.pt", weights_only=False)
        print("[*] 成功加载原始实验数据，开始进行学术级精修...")
    except FileNotFoundError:
        print("[!] 错误：找不到原始数据文件（.pt）。请确保你已经运行过 exp_adversary.py。")
        return

    # ========================== 绘图 A: 知识等级 (精修折线图) ==========================
    plt.figure(figsize=(10, 6))
    epochs_x = np.arange(len(results_knowledge['COA (0-bit Key)'])) * 100
    # 选取更具学术感的深色调配色
    colors = ['#1f77b4', '#ff7f0e', '#d62728']
    
    for (label, acc_list), col in zip(results_knowledge.items(), colors):
        # 使用滑动平均进行平滑处理，减少随机震荡，突出趋势
        smoothed = np.convolve(acc_list, np.ones(10)/10, mode='valid') * 100
        plt.plot(epochs_x[:len(smoothed)], smoothed, label=label, color=col, linewidth=2.5)
        
    plt.axhline(y=50, color='gray', linestyle='--', linewidth=1.5, label='Random Guess (50%)')
    
    plt.title('Knowledge Hierarchy Analysis: Resilience to Partial Key Leakage', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Training Epochs', fontsize=12, fontweight='bold')
    plt.ylabel('Eve Decryption Accuracy (%)', fontsize=12, fontweight='bold')
    
    # --- 专项优化：解决不匀称感 ---
    # 将 Y 轴范围从 [45, 105] 压缩到 [48, 75]
    # 这会产生“局部放大”效果，展示 COA 到 PKE-8bit 之间那关键的几个百分点的防御差异
    plt.ylim([48, 75]) 
    
    # 调整图例位置，避免遮挡曲线
    plt.legend(loc='upper left', fontsize=10, frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("exp_knowledge_hierarchy_fixed.png", dpi=300)
    print("[+] 实验A精修图已保存：exp_knowledge_hierarchy_fixed.png")

    # ========================== 绘图 B: 对抗容量 (精修趋势图) ==========================
    plt.figure(figsize=(9, 6.5))
    depths = results_capacity['depths']
    bob_accs = results_capacity['bob_accs']
    eve_accs = results_capacity['eve_accs']

    # 绘制核心趋势线
    line_bob, = plt.plot(depths, bob_accs, marker='o', markersize=10, color='#2ca02c', linewidth=3, label='Bob (Receiver)')
    line_eve, = plt.plot(depths, eve_accs, marker='s', markersize=10, color='#d62728', linewidth=3, label='Eve (Adversary)')
    
    # --- 专项优化 1：添加安全间隙阴影 ---
    plt.fill_between(depths, eve_accs, bob_accs, color='#1f77b4', alpha=0.12, label='Security Margin (Gap)')
    
    # --- 专项优化 2：标注每个点的具体数值 ---
    # 使用不同的垂直偏量，确保标注不会重叠
    for i, (b, e) in enumerate(zip(bob_accs, eve_accs)):
        plt.text(depths[i], b + 2.5, f'{b:.1f}%', ha='center', va='bottom', color='#2ca02c', fontsize=10, fontweight='bold')
        plt.text(depths[i], e - 4.5, f'{e:.1f}%', ha='center', va='top', color='#d62728', fontsize=10, fontweight='bold')

    plt.title('Adversary Capacity Scaling: Robustness vs Network Depth', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Eve Network Depth (ResNet Blocks)', fontsize=12, fontweight='bold')
    plt.ylabel('Final Stable Accuracy (%)', fontsize=12, fontweight='bold')
    plt.xticks(depths)
    
    # 稍微拉高 Y 轴上限，为顶部的数值标注留出空间
    plt.ylim([40, 115]) 
    
    plt.axhline(y=50, color='gray', linestyle='--', linewidth=1.5, label='Random Baseline')
    plt.legend(loc='center right', fontsize=10, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig("exp_capacity_scaling_fixed.png", dpi=300)
    print("[+] 实验B精修图已保存：exp_capacity_scaling_fixed.png")

if __name__ == "__main__":
    redraw_final_plots()