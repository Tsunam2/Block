import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from gan_engine import GANCryptoEngine, set_seed
import os

def run_fidelity_test():
    # 1. 环境准备
    set_seed(42)
    engine = GANCryptoEngine(msg_len=16, key_len=16)
    checkpoint_path = "best_checkpoint.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"[!] 错误：找不到 {checkpoint_path}。请确保已经运行过主训练脚本。")
        return

    # 2. 载入巅峰权重
    print(f"[*] 正在载入 {checkpoint_path} 进行忠实度评估...")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    engine.alice.load_state_dict(checkpoint['alice'])
    engine.bob.load_state_dict(checkpoint['bob'])
    engine.alice.eval()
    engine.bob.eval()

    # 3. 生成大规模测试数据 (10万比特)
    num_batches = 400 
    batch_size = 256
    all_targets = []
    all_recovered = []

    with torch.no_grad():
        for _ in range(num_batches):
            # 生成原始明文 (0 或 1)
            p_raw = torch.randint(0, 2, (batch_size, engine.msg_len)).float()
            p_input = p_raw * 2 - 1 # 映射到 [-1, 1] 喂给网络
            pk = torch.randn(batch_size, engine.key_len)
            
            # 经过 Alice 和 Bob
            c = engine.alice(p_input, pk)
            recovered_raw = engine.bob(c, pk)
            
            # 解码回比特位 (0 或 1)
            p_recovered = (recovered_raw > 0).float()
            
            all_targets.extend(p_raw.view(-1).numpy())
            all_recovered.extend(p_recovered.view(-1).numpy())

    # 4. 计算混淆矩阵并归一化为概率
    cm = confusion_matrix(all_targets, all_recovered)
    cm_prob = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # ========================== 绘制学术混淆矩阵 ==========================
    plt.figure(figsize=(7, 6))
    
    # 使用自定义调色板：深蓝色代表高概率
    sns.heatmap(cm_prob, annot=True, fmt=".4f", cmap='Blues', 
                xticklabels=['Decoded 0', 'Decoded 1'], 
                yticklabels=['Original 0', 'Original 1'],
                annot_kws={"size": 14, "weight": "bold"},
                cbar_kws={'label': 'Transition Probability'})

    plt.title('Bit-wise Decryption Fidelity (Bob\'s Confusion Matrix)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Predicted Bit (Bob Output)', fontsize=12, fontweight='bold')
    plt.ylabel('Ground Truth Bit (Plaintext)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("exp_confusion_matrix.png", dpi=300)
    print("[*] 忠实度实验完成！混淆矩阵已保存为: exp_confusion_matrix.png")
    
    # 打印具体统计
    print("-" * 30)
    print(f"统计样本量: {len(all_targets)} bits")
    print(f"0 -> 0 成功率: {cm_prob[0,0]:.4%}")
    print(f"1 -> 1 成功率: {cm_prob[1,1]:.4%}")
    print("-" * 30)

if __name__ == "__main__":
    run_fidelity_test()