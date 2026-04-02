import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
import random

# =================================================================
# 0. 全局随机种子固定 (确保学术复现性)
# =================================================================
def set_seed(seed=42):
    """强制锁死所有的随机性，保证每次运行结果绝对一致"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =================================================================
# 1. 网络架构定义 (严格保持原版复杂度)
# =================================================================

class CryptoBlock(nn.Module):
    """基础残差/线性块，确保特征提取深度"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class AliceNet(nn.Module):
    """Alice: 负责将明文和密钥混合并加密"""
    def __init__(self, msg_len, key_len):
        super().__init__()
        self.fc = nn.Linear(msg_len + key_len, 512)
        self.blocks = nn.Sequential(
            CryptoBlock(512, 512),
            CryptoBlock(512, 512),
            CryptoBlock(512, 512)
        )
        self.output = nn.Sequential(
            nn.Linear(512, msg_len),
            nn.Tanh()
        )
    def forward(self, p, pk):
        x = torch.cat((p, pk), dim=1)
        x = self.fc(x)
        x = self.blocks(x)
        return self.output(x)

class BobNet(nn.Module):
    """Bob: 负责根据密文和密钥解密"""
    def __init__(self, msg_len, key_len):
        super().__init__()
        self.fc = nn.Linear(msg_len + key_len, 512)
        self.blocks = nn.Sequential(
            CryptoBlock(512, 512),
            CryptoBlock(512, 512),
            CryptoBlock(512, 512)
        )
        self.output = nn.Sequential(
            nn.Linear(512, msg_len),
            nn.Tanh()
        )
    def forward(self, c, sk):
        x = torch.cat((c, sk), dim=1)
        x = self.fc(x)
        x = self.blocks(x)
        return self.output(x)

class EveNet(nn.Module):
    """Eve: 攻击者，尝试仅从密文恢复明文"""
    def __init__(self, msg_len):
        super().__init__()
        self.fc = nn.Linear(msg_len, 512)
        self.blocks = nn.Sequential(
            CryptoBlock(512, 512),
            CryptoBlock(512, 1024),
            CryptoBlock(1024, 1024),
            CryptoBlock(1024, 512)
        )
        self.output = nn.Sequential(
            nn.Linear(512, msg_len),
            nn.Tanh()
        )
    def forward(self, c):
        x = self.fc(c)
        x = self.blocks(x)
        return self.output(x)

# =================================================================
# 2. 训练引擎 (集成“保存最佳”策略与优化可视化)
# =================================================================

class GANCryptoEngine:
    def __init__(self, msg_len=16, key_len=16, lr=0.0008):
        self.msg_len = msg_len
        self.key_len = key_len
        
        # 初始化网络
        self.alice = AliceNet(msg_len, key_len)
        self.bob = BobNet(msg_len, key_len)
        self.eve = EveNet(msg_len)
        
        # 损失函数
        self.criterion = nn.L1Loss()
        self.criterion_sum = nn.L1Loss(reduction='sum')
        
        # 优化器
        self.opt_ab = optim.Adam(list(self.alice.parameters()) + list(self.bob.parameters()), lr=lr)
        self.opt_e = optim.Adam(self.eve.parameters(), lr=lr)
        
        # 补齐 10000 轮的衰减节点
        self.scheduler_ab = optim.lr_scheduler.MultiStepLR(self.opt_ab, milestones=[5000, 8000, 10000], gamma=0.5)
        self.scheduler_e = optim.lr_scheduler.MultiStepLR(self.opt_e, milestones=[5000, 8000, 10000], gamma=0.5)
        
        # 数据统计
        self.stats = {
            'bob_l1': [], 'eve_l1': [], 'penalty': [],
            'bob_acc': [], 'eve_acc': [],
            'bob_bitwise_acc': []  
        }
        
        # 保存最佳策略相关变量
        self.best_bob_acc = 0.0
        self.security_threshold = 0.60 # Eve 准确率低于 60% 才视为安全

    def encrypt(self, raw_bits, key_vector):
        """Alice 加密接口"""
        self.alice.eval()
        with torch.no_grad():
            p = torch.FloatTensor(raw_bits).view(1, -1) * 2 - 1
            pk = torch.FloatTensor(key_vector).view(1, -1)
            cipher_vec = self.alice(p, pk)
            return cipher_vec.squeeze().numpy()

    def decrypt(self, cipher_vec, key_vector):
        """Bob 解密接口"""
        self.bob.eval()
        with torch.no_grad():
            c = torch.FloatTensor(cipher_vec).view(1, -1)
            sk = torch.FloatTensor(key_vector).view(1, -1)
            recovered = self.bob(c, sk)
            bits = (recovered > 0).int().squeeze().numpy()
            return bits

    def _get_acc(self, pred, target):
        """计算比特级准确率"""
        with torch.no_grad():
            p_bits = (pred > 0).float()
            t_bits = (target > 0).float()
            return (p_bits == t_bits).float().mean().item()

    def train(self, epochs=15000, batch_size=256):
        print(f"[GAN] 引擎启动 | 目标轮数: {epochs} | 消息长度: {self.msg_len}")
        
        for epoch in range(epochs):
            p = torch.randint(0, 2, (batch_size, self.msg_len)).float() * 2 - 1
            pk = torch.randn(batch_size, self.key_len)
            
            # --- 训练 Eve (攻击者) ---
            for _ in range(2):
                self.opt_e.zero_grad()
                with torch.no_grad():
                    c = self.alice(p, pk)
                p_eve = self.eve(c)
                l_eve = self.criterion(p_eve, p)
                l_eve.backward()
                self.opt_e.step()

            # --- 训练 Alice & Bob ---
            for _ in range(2):
                self.opt_ab.zero_grad()
                c_new = self.alice(p, pk)
                p_bob = self.bob(c_new, pk)
                p_eve_adv = self.eve(c_new)
                
                l_rec = self.criterion(p_bob, p)
                l_eve_adv_sum = self.criterion_sum(p_eve_adv, p) / batch_size
                
                half_n = float(self.msg_len) 
                penalty = torch.pow(torch.clamp(half_n - l_eve_adv_sum, min=0), 2) / (half_n ** 2)
                
                total_loss = l_rec + 3.0 * penalty
                
                total_loss.backward()
                self.opt_ab.step()
            
            self.scheduler_ab.step()
            self.scheduler_e.step()
            
            # 记录历史数据 
            if epoch % 10 == 0:
                b_acc = self._get_acc(p_bob, p)
                e_acc = self._get_acc(p_eve, p)
                self.stats['bob_l1'].append(l_rec.item())
                self.stats['eve_l1'].append(l_eve.item())
                self.stats['penalty'].append(penalty.item())
                self.stats['bob_acc'].append(b_acc)
                self.stats['eve_acc'].append(e_acc)
                
                # 记录逐位准确率
                with torch.no_grad():
                    p_bob_bits = (p_bob > 0).float()
                    p_bits = (p > 0).float()
                    bitwise_match = (p_bob_bits == p_bits).float()
                    bitwise_acc = bitwise_match.mean(dim=0).cpu().numpy() 
                    self.stats['bob_bitwise_acc'].append(bitwise_acc)

                # --- 保存最佳策略 ---
                if b_acc > self.best_bob_acc and e_acc < self.security_threshold:
                    self.best_bob_acc = b_acc
                    self.save_checkpoint("best_checkpoint.pth")
                    print(f"  >>> [最佳平衡点] Epoch {epoch} | Bob Acc 创新高: {b_acc:.2%} (Eve Acc: {e_acc:.2%})")

            # 终端输出控制
            if epoch % 500 == 0 or epoch == epochs - 1:
                curr_b_acc = self._get_acc(p_bob, p)
                curr_e_acc = self._get_acc(p_eve, p)
                curr_lr = self.opt_ab.param_groups[0]['lr']
                print(f"Epoch {epoch:5d} | "
                      f"Bob L1: {l_rec.item():.4f} | "
                      f"Eve L1: {l_eve.item():.4f} | "
                      f"Pen: {penalty.item():.4f} | "
                      f"Bob Acc: {curr_b_acc:.2%} | "
                      f"Eve Acc: {curr_e_acc:.2%} | "
                      f"LR: {curr_lr:.6f}")

    def save_checkpoint(self, filename="checkpoint.pth"):
        state = {
            'alice': self.alice.state_dict(),
            'bob': self.bob.state_dict(),
            'eve': self.eve.state_dict(),
            'stats': self.stats
        }
        torch.save(state, filename)

    def visualize(self):
        import matplotlib.patches as mpatches
        
        steps = np.arange(len(self.stats['bob_l1'])) * 10
        
        # ==========================================
        # 图 1：双 Y 轴安全间隙图 (Security Gap Plot)
        # ==========================================
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        
        # 左侧 Y 轴 (准确率)
        ax1.set_xlabel('Training Epochs', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Reconstruction Accuracy', fontsize=12, fontweight='bold')
        ax1.set_ylim([0.0, 1.05])
        
        line_bob, = ax1.plot(steps, self.stats['bob_acc'], color='#2ca02c', linewidth=2.5, label='Bob Acc (Receiver)')
        line_eve, = ax1.plot(steps, self.stats['eve_acc'], color='#d62728', linewidth=2.5, label='Eve Acc (Adversary)')
        
        # 填充核心阴影区
        ax1.fill_between(steps, self.stats['eve_acc'], self.stats['bob_acc'], 
                         where=(np.array(self.stats['bob_acc']) > np.array(self.stats['eve_acc'])),
                         color='#1f77b4', alpha=0.15)
        
        # 右侧 Y 轴 (独立的 L1 Loss)
        ax2 = ax1.twinx()
        ax2.set_ylabel('Reconstruction Loss (L1)', color='#7f7f7f', fontsize=12, fontweight='bold')
        
        # 强制拉高 Loss Y轴的上限
        ax2.set_ylim([0, 1.25])
        
        line_bob_loss, = ax2.plot(steps, self.stats['bob_l1'], color='royalblue', linestyle='--', linewidth=1.5, label='Bob Loss')
        line_eve_loss, = ax2.plot(steps, self.stats['eve_l1'], color='tomato', linestyle='--', linewidth=1.5, label='Eve Loss')
        ax2.tick_params(axis='y', labelcolor='#7f7f7f')
        
        # 合并图例并调整位置
        patch_gain = mpatches.Patch(color='#1f77b4', alpha=0.15, label='Security Gain')
        ax1.legend(handles=[line_bob, line_eve, patch_gain, line_bob_loss, line_eve_loss], 
                   bbox_to_anchor=(0.98, 0.35), loc='center right', fontsize=10)
        
        plt.title('Training Dynamics & Security Margin', fontsize=14, fontweight='bold', pad=15)
        ax1.grid(True, linestyle=':', alpha=0.6)
        
        fig1.tight_layout()
        fig1.savefig("security_gap_analysis.png", dpi=300)
        print("[*] 图1：双Y轴安全间隙图已保存为: security_gap_analysis.png")
        
        # ==========================================
        # 图 2：训练状态热力图 (Bit-wise Heatmap)
        # ==========================================
        fig2, ax_heat = plt.subplots(figsize=(10, 6))
        
        heatmap_data = np.array(self.stats['bob_bitwise_acc']).T
        
        cax = ax_heat.imshow(heatmap_data, aspect='auto', cmap='viridis', 
                             origin='lower', extent=[0, steps[-1], 0, self.msg_len - 1])
        
        ax_heat.set_xlabel('Training Epochs', fontsize=12, fontweight='bold')
        ax_heat.set_ylabel('Message Bit Index', fontsize=12, fontweight='bold')
        ax_heat.set_yticks(np.arange(0, self.msg_len))
        
        cbar = fig2.colorbar(cax, ax=ax_heat)
        cbar.set_label('Bob Decryption Accuracy', rotation=270, labelpad=20, fontsize=12)
        
        plt.title('Bit-wise Decryption Evolution (Heatmap)', fontsize=14, fontweight='bold', pad=15)
        
        fig2.tight_layout()
        fig2.savefig("bitwise_heatmap.png", dpi=300)
        print("[*] 图2：训练状态热力图已保存为: bitwise_heatmap.png")

# --- Main 函数 ---
if __name__ == "__main__":
    # 虽然不训练，但也习惯性加上种子
    set_seed(42) 
    
    engine = GANCryptoEngine(msg_len=16, key_len=16)
    
    # 核心操作：不调用 engine.train()，而是直接加载刚刚跑出的最佳模型
    checkpoint_path = "best_checkpoint.pth"
    if os.path.exists(checkpoint_path):
        print(f"[*] 发现 {checkpoint_path}，正在载入巅峰数据...")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        
        # 恢复网络权重 (虽然画图用不到权重，但严谨起见一并加载)
        engine.alice.load_state_dict(checkpoint['alice'])
        engine.bob.load_state_dict(checkpoint['bob'])
        engine.eve.load_state_dict(checkpoint['eve'])
        
        # 【最关键的一步】：恢复统计数据
        engine.stats = checkpoint['stats']
        
        print("[*] 数据载入成功！正在重新生成图表...")
        # 直接调用画图
        engine.visualize()
    else:
        print(f"[!] 找不到 {checkpoint_path}，请确认文件是否存在。")