import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt

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
# 2. 训练引擎 (增强可视化与输出逻辑)
# =================================================================

class GANCryptoEngine:
    def __init__(self, msg_len=16, key_len=16, lr=0.0008):
        self.msg_len = msg_len
        self.key_len = key_len
        
        # 初始化网络
        self.alice = AliceNet(msg_len, key_len)
        self.bob = BobNet(msg_len, key_len)
        self.eve = EveNet(msg_len)
        
        # 损失函数 (绝对不允许修改定义)
        self.criterion = nn.L1Loss()
        self.criterion_sum = nn.L1Loss(reduction='sum') # 用于计算惩罚项
        
        # 优化器
        self.opt_ab = optim.Adam(list(self.alice.parameters()) + list(self.bob.parameters()), lr=lr)
        self.opt_e = optim.Adam(self.eve.parameters(), lr=lr)
        
        # --- 方案 1 修改点：引入学习率调度器 ---
        # 在 5000 轮和 8000 轮时分别将学习率降低为原来的 0.5 倍
        self.scheduler_ab = optim.lr_scheduler.MultiStepLR(self.opt_ab, milestones=[5000, 8000], gamma=0.5)
        self.scheduler_e = optim.lr_scheduler.MultiStepLR(self.opt_e, milestones=[5000, 8000], gamma=0.5)
        
        # 数据统计
        self.stats = {
            'bob_l1': [], 'eve_l1': [], 'penalty': [],
            'bob_acc': [], 'eve_acc': []
        }

    # ========================== 新增接口 ==========================
    def encrypt(self, raw_bits, key_vector):
        """Alice 加密接口"""
        self.alice.eval()
        with torch.no_grad():
            # 内部自动映射 0/1 到 -1/1
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
            # 还原比特流
            bits = (recovered > 0).int().squeeze().numpy()
            return bits
    # =============================================================

    def _get_acc(self, pred, target):
        """计算比特级准确率"""
        with torch.no_grad():
            p_bits = (pred > 0).float()
            t_bits = (target > 0).float()
            return (p_bits == t_bits).float().mean().item()

    def train(self, epochs=10000, batch_size=256):
        print(f"[GAN] 引擎启动 | 消息长度: {self.msg_len} | 密钥长度: {self.key_len}")
        
        for epoch in range(epochs):
            # 1. 生成随机明文 P 和 密钥 K
            p = torch.randint(0, 2, (batch_size, self.msg_len)).float() * 2 - 1
            pk = torch.randn(batch_size, self.key_len)
            
            # ---------------------
            #   训练 Eve (攻击者) - 保持训练频率 2 次
            # ---------------------
            for _ in range(2):
                self.opt_e.zero_grad()
                with torch.no_grad():
                    c = self.alice(p, pk)
                p_eve = self.eve(c)
                l_eve = self.criterion(p_eve, p)
                l_eve.backward()
                self.opt_e.step()

            # ---------------------
            #   训练 Alice & Bob - 保持训练频率 2 次
            # ---------------------
            for _ in range(2):
                self.opt_ab.zero_grad()
                c_new = self.alice(p, pk)
                p_bob = self.bob(c_new, pk)
                p_eve_adv = self.eve(c_new)
                
                l_rec = self.criterion(p_bob, p)
                l_eve_adv_sum = self.criterion_sum(p_eve_adv, p) / batch_size
                
                # 保持 half_n 为 msg_len (即 16.0)，对应平均 L1 = 1.0 的安全目标
                half_n = float(self.msg_len) 
                
                # 惩罚项公式保持不变
                penalty = torch.pow(torch.clamp(half_n - l_eve_adv_sum, min=0), 2) / (half_n ** 2)
                
                # 维持方案 A 的权重 3.0
                total_loss = l_rec + 3.0 * penalty
                
                total_loss.backward()
                self.opt_ab.step()
            
            # --- 方案 1 修改点：更新学习率 ---
            self.scheduler_ab.step()
            self.scheduler_e.step()
            
            # 记录历史数据 (每10轮记录一次)
            if epoch % 10 == 0:
                b_acc = self._get_acc(p_bob, p)
                e_acc = self._get_acc(p_eve, p)
                self.stats['bob_l1'].append(l_rec.item())
                self.stats['eve_l1'].append(l_eve.item())
                self.stats['penalty'].append(penalty.item())
                self.stats['bob_acc'].append(b_acc)
                self.stats['eve_acc'].append(e_acc)

            # 终端输出控制 (严格保持原版输出格式)
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
        """保存完整模型和训练统计"""
        state = {
            'alice': self.alice.state_dict(),
            'bob': self.bob.state_dict(),
            'eve': self.eve.state_dict(),
            'stats': self.stats
        }
        torch.save(state, filename)
        print(f"[*] 模型已保存至 {filename}")

    def visualize(self):
        """生成并保存分析图表 (严格保持原版绘图逻辑)"""
        steps = np.arange(len(self.stats['bob_l1'])) * 10
        
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        mask_1000 = steps <= 1000
        plt.plot(steps[mask_1000], np.array(self.stats['bob_l1'])[mask_1000], label='Bob L1')
        plt.plot(steps[mask_1000], np.array(self.stats['eve_l1'])[mask_1000], label='Eve L1')
        plt.title("Loss Trends (First 1000 Epochs)")
        plt.xlabel("Epochs"); plt.ylabel("L1 Loss"); plt.legend(); plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(steps, self.stats['bob_l1'], label='Bob L1')
        plt.plot(steps, self.stats['eve_l1'], label='Eve L1')
        plt.title("Loss Trends (Full Process)")
        plt.xlabel("Epochs"); plt.ylabel("L1 Loss"); plt.legend(); plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("loss_analysis.png", dpi=300)
        print("[*] 损失曲线图已保存为: loss_analysis.png")

        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        plt.plot(steps[mask_1000], np.array(self.stats['bob_acc'])[mask_1000], label='Bob Accuracy', color='green')