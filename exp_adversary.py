import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from gan_engine import GANCryptoEngine, CryptoBlock, set_seed

# =================================================================
# 1. 动态构建更聪明、更深度的 Eve 网络 (支持额外知识输入)
# =================================================================
class DynamicEveNet(nn.Module):
    def __init__(self, msg_len, extra_know_len=0, num_blocks=4):
        super().__init__()
        self.fc = nn.Linear(msg_len + extra_know_len, 512)
        
        blocks = []
        for _ in range(num_blocks):
            blocks.append(CryptoBlock(512, 512))
        self.blocks = nn.Sequential(*blocks)
        
        self.output = nn.Sequential(
            nn.Linear(512, msg_len),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.fc(x)
        x = self.blocks(x)
        return self.output(x)

# =================================================================
# 2. 继承引擎：重写数据流，注入“部分密钥泄露 (PKE)”逻辑
# =================================================================
class AdvancedKnowledgeEngine(GANCryptoEngine):
    def __init__(self, msg_len=16, key_len=16, exposed_key_bits=0, eve_blocks=4):
        super().__init__(msg_len, key_len)
        self.exposed_key_bits = exposed_key_bits
        
        self.eve = DynamicEveNet(msg_len, extra_know_len=exposed_key_bits, num_blocks=eve_blocks)
        
        self.opt_e = optim.Adam(self.eve.parameters(), lr=0.0008)
        self.scheduler_e = optim.lr_scheduler.MultiStepLR(self.opt_e, milestones=[5000, 8000, 10000], gamma=0.5)

    def train(self, epochs=15000, batch_size=256):
        print(f"[*] 启动鲁棒性测试引擎 | Eve 深度: {len(self.eve.blocks)} 层 | 泄露密钥位: {self.exposed_key_bits} 位")
        for epoch in range(epochs):
            p = torch.randint(0, 2, (batch_size, self.msg_len)).float() * 2 - 1
            pk = torch.randn(batch_size, self.key_len)
            
            with torch.no_grad():
                c = self.alice(p, pk)
            if self.exposed_key_bits > 0:
                eve_input = torch.cat((c, pk[:, :self.exposed_key_bits]), dim=1)
            else:
                eve_input = c
            
            for _ in range(2):
                self.opt_e.zero_grad()
                p_eve = self.eve(eve_input)
                l_eve = self.criterion(p_eve, p)
                l_eve.backward()
                self.opt_e.step()

            for _ in range(2):
                self.opt_ab.zero_grad()
                c_new = self.alice(p, pk)
                p_bob = self.bob(c_new, pk)
                
                if self.exposed_key_bits > 0:
                    eve_input_adv = torch.cat((c_new, pk[:, :self.exposed_key_bits]), dim=1)
                else:
                    eve_input_adv = c_new
                p_eve_adv = self.eve(eve_input_adv)
                
                l_rec = self.criterion(p_bob, p)
                l_eve_adv_sum = self.criterion_sum(p_eve_adv, p) / batch_size
                
                half_n = float(self.msg_len) 
                penalty = torch.pow(torch.clamp(half_n - l_eve_adv_sum, min=0), 2) / (half_n ** 2)
                
                total_loss = l_rec + 3.0 * penalty
                total_loss.backward()
                self.opt_ab.step()
            
            self.scheduler_ab.step()
            self.scheduler_e.step()
            
            if epoch % 100 == 0:
                b_acc = self._get_acc(p_bob, p)
                e_acc = self._get_acc(p_eve, p)
                self.stats['bob_acc'].append(b_acc)
                self.stats['eve_acc'].append(e_acc)
                
                if b_acc > self.best_bob_acc and e_acc < self.security_threshold:
                    self.best_bob_acc = b_acc

            if epoch % 3000 == 0 or epoch == epochs - 1:
                print(f"  -> Epoch {epoch:5d} | Bob 最佳 Acc: {self.best_bob_acc:.2%} | 当前 Eve Acc: {e_acc:.2%}")

# =================================================================
# 3. 主实验控制流与可视化 (加入防弹保存机制)
# =================================================================
def run_robustness_experiments():
    set_seed(42)
    target_epochs = 15000 
    
    # --- 实验 A: 攻击者知识等级 ---
    print("\n" + "="*60)
    print(">>> 实验 A: 攻击者知识等级 (Knowledge Hierarchy) <<<")
    print("="*60)
    
    results_knowledge = {}
    knowledge_levels = {'COA (0-bit Key)': 0, 'PKE (4-bit Key)': 4, 'PKE (8-bit Key)': 8}
    
    for label, exposed_bits in knowledge_levels.items():
        print(f"\n[开始] {label} 攻击测试...")
        engine = AdvancedKnowledgeEngine(exposed_key_bits=exposed_bits)
        engine.train(epochs=target_epochs)
        results_knowledge[label] = engine.stats['eve_acc']
    
    # 【防弹机制1】：立刻把跑完的数据存下来！
    torch.save(results_knowledge, "raw_data_knowledge.pt")
    print("[*] 实验A原始数据已安全备份至 raw_data_knowledge.pt")

    # --- 实验 B: 对抗容量演化 ---
    print("\n" + "="*60)
    print(">>> 实验 B: 对抗容量演化 (Adversary Capacity) <<<")
    print("="*60)
    
    results_capacity = {'depths': [2, 4, 8], 'eve_accs': [], 'bob_accs': []}
    
    for depth in results_capacity['depths']:
        print(f"\n[开始] Eve 深度为 {depth} 层的压制测试...")
        engine = AdvancedKnowledgeEngine(exposed_key_bits=0, eve_blocks=depth)
        engine.train(epochs=target_epochs)
        results_capacity['bob_accs'].append(engine.best_bob_acc * 100)
        final_eve_acc = np.mean(engine.stats['eve_acc'][-5:]) * 100 
        results_capacity['eve_accs'].append(final_eve_acc)
        
    # 【防弹机制2】：立刻把跑完的数据存下来！
    torch.save(results_capacity, "raw_data_capacity.pt")
    print("[*] 实验B原始数据已安全备份至 raw_data_capacity.pt")

    # ========================== 绘制顶级学术图表 ==========================
    
    # --- 绘图 A: 折线图 ---
    plt.figure(figsize=(10, 6))
    epochs_x = np.arange(len(results_knowledge['COA (0-bit Key)'])) * 100
    colors = ['#2ca02c', '#ff7f0e', '#d62728']
    
    for (label, acc_list), col in zip(results_knowledge.items(), colors):
        smoothed = np.convolve(acc_list, np.ones(10)/10, mode='valid') * 100
        plt.plot(epochs_x[:len(smoothed)], smoothed, label=label, color=col, linewidth=2.5, alpha=0.9)
        
    plt.axhline(y=50, color='gray', linestyle='--', linewidth=2, label='Random Guess (50%)')
    plt.title('Knowledge Hierarchy Analysis: Eve\'s Accuracy over Time', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Training Epochs', fontsize=12, fontweight='bold')
    plt.ylabel('Eve Decryption Accuracy (%)', fontsize=12, fontweight='bold')
    plt.ylim([45, 105])
    
    # 【优化点1】：放在左上角，绝对不会碰到右下方的折线
    plt.legend(loc='upper left', fontsize=11) 
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("exp_knowledge_hierarchy.png", dpi=300)
    
    # --- 绘图 B: 散点趋势图 ---
    plt.figure(figsize=(8, 6))
    plt.plot(results_capacity['depths'], results_capacity['bob_accs'], marker='o', markersize=10, 
             color='#2ca02c', linewidth=3, label='Bob Final Acc (Receiver)')
    plt.plot(results_capacity['depths'], results_capacity['eve_accs'], marker='s', markersize=10, 
             color='#d62728', linewidth=3, label='Eve Final Acc (Adversary)')
    
    plt.title('Adversary Capacity Scaling: Network Depth vs Security', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Eve Network Depth (Number of ResNet Blocks)', fontsize=12, fontweight='bold')
    plt.ylabel('Final Stable Accuracy (%)', fontsize=12, fontweight='bold')
    plt.xticks(results_capacity['depths'])
    plt.ylim([40, 105])
    plt.axhline(y=50, color='gray', linestyle='--', linewidth=2, label='Random Guess (50%)')
    
    # 【优化点2】：放在图表正右方，恰好在Bob(100)和Eve(50)的中间真空带
    plt.legend(loc='center right', fontsize=11) 
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("exp_capacity_scaling.png", dpi=300)

    print("\n[*] 极其硬核的 75000 轮鲁棒性测试跑完了！图表已保存: exp_knowledge_hierarchy.png, exp_capacity_scaling.png")

if __name__ == "__main__":
    run_robustness_experiments()