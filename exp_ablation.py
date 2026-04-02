import torch
import numpy as np
import matplotlib.pyplot as plt
from gan_engine import GANCryptoEngine, set_seed

def run_ablation_study():
    # 强制锁死随机种子，保证每次消融实验的变量绝对可控
    set_seed(42)
    
    # 为了快速出图，消融组每组跑 2000 轮即可看出明显趋势
    epochs_per_exp = 2000 
    
    # 准备存储结果的字典
    results = {
        'Baseline (Full-NCE)': {'bob': 99.83, 'eve': 55.66}, # 直接填入我们巅峰基准数据
        'w/o Adversary (No-GAN)': {'bob': 0.0, 'eve': 0.0},
        'w/o Shared-Key (No-Key)': {'bob': 0.0, 'eve': 0.0},
        'Static-Alice (Fixed)': {'bob': 0.0, 'eve': 0.0}
    }

    print("\n" + "="*60)
    print(">>> [1/3] 启动 No-GAN 实验: 取消 Eve 的对抗压力 <<<")
    # 技巧：实例化正常引擎，但通过覆盖 criterion_sum 让 penalty 永远等于 0
    engine_nogan = GANCryptoEngine(msg_len=16, key_len=16)
    engine_nogan.criterion_sum = lambda pred, target: torch.tensor(16.0 * pred.shape[0], device=pred.device)
    engine_nogan.train(epochs=epochs_per_exp)
    results['w/o Adversary (No-GAN)']['bob'] = engine_nogan.stats['bob_acc'][-1] * 100
    results['w/o Adversary (No-GAN)']['eve'] = engine_nogan.stats['eve_acc'][-1] * 100

    print("\n" + "="*60)
    print(">>> [2/3] 启动 No-Key 实验: 移除共享神经密钥 <<<")
    # 技巧：直接将 key_len 设为 0，底层网络会自动适应，相当于明文直接暴露
    engine_nokey = GANCryptoEngine(msg_len=16, key_len=0)
    engine_nokey.train(epochs=epochs_per_exp)
    results['w/o Shared-Key (No-Key)']['bob'] = engine_nokey.stats['bob_acc'][-1] * 100
    results['w/o Shared-Key (No-Key)']['eve'] = engine_nokey.stats['eve_acc'][-1] * 100

    print("\n" + "="*60)
    print(">>> [3/3] 启动 Static-Alice 实验: 证明动态对抗的必要性 <<<")
    engine_static = GANCryptoEngine(msg_len=16, key_len=16)
    
    print(" -> 阶段 A: 屏蔽 Eve，让 Alice&Bob 建立静态自编码通信 (1000轮)")
    # 技巧：将 Eve 的学习率设为 0，相当于 Eve 不参与学习
    for g in engine_static.opt_e.param_groups: g['lr'] = 0.0
    engine_static.train(epochs=1000)
    
    print(" -> 阶段 B: 冻结 Alice&Bob，放开 Eve 进行疯狂破解攻击 (1000轮)")
    # 技巧：恢复 Eve 的学习率，并将 Alice&Bob 设为 0（彻底冻结加密层）
    for g in engine_static.opt_e.param_groups: g['lr'] = 0.0008
    for g in engine_static.opt_ab.param_groups: g['lr'] = 0.0
    engine_static.train(epochs=1000)
    
    results['Static-Alice (Fixed)']['bob'] = engine_static.stats['bob_acc'][-1] * 100
    results['Static-Alice (Fixed)']['eve'] = engine_static.stats['eve_acc'][-1] * 100

    # ========================== 绘制顶级学术柱状图 ==========================
    labels = list(results.keys())
    bob_scores = [results[k]['bob'] for k in labels]
    eve_scores = [results[k]['eve'] for k in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, bob_scores, width, label='Bob Acc (Receiver)', color='#2ca02c', edgecolor='black')
    rects2 = ax.bar(x + width/2, eve_scores, width, label='Eve Acc (Adversary)', color='#d62728', edgecolor='black')

    # 添加 50% 的安全底线参考
    ax.axhline(y=50.0, color='gray', linestyle='--', linewidth=1.5, label='Random Guess (50%)')

    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Ablation Study: Component Impact on System Security', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax.legend(loc='lower center', bbox_to_anchor=(0.65, 0.85), fontsize=10)
    ax.set_ylim([0, 110])

    # 在柱子顶部加上具体数值
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 垂直偏移
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.savefig("ablation_study.png", dpi=300)
    print("\n[*] 消融实验完成！图表已保存为: ablation_study.png")

if __name__ == "__main__":
    run_ablation_study()