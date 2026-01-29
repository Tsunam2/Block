import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

# =================================================================
# 1. 强化版架构设计 (保持不变，确保兼容性)
# =================================================================

class CryptoBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):
        return self.net(x)

class AliceNet(nn.Module):
    def __init__(self, msg_len, key_len):
        super().__init__()
        self.main = nn.Sequential(
            CryptoBlock(msg_len + key_len, 512),
            CryptoBlock(512, 512),
            nn.Linear(512, msg_len),
            nn.Tanh()
        )
    def forward(self, p, pk):
        return self.main(torch.cat((p, pk), dim=1))

class BobNet(nn.Module):
    def __init__(self, msg_len, key_len):
        super().__init__()
        self.main = nn.Sequential(
            CryptoBlock(msg_len + key_len, 512),
            CryptoBlock(512, 512),
            nn.Linear(512, msg_len),
            nn.Tanh()
        )
    def forward(self, c, sk):
        return self.main(torch.cat((c, sk), dim=1))

class EveNet(nn.Module):
    def __init__(self, msg_len):
        super().__init__()
        self.main = nn.Sequential(
            CryptoBlock(msg_len, 512),
            CryptoBlock(512, 512),
            nn.Linear(512, msg_len),
            nn.Tanh()
        )
    def forward(self, c):
        return self.main(c)

# =================================================================
# 2. 核心引擎与优化训练逻辑
# =================================================================

class GANCryptoEngine:
    def __init__(self, msg_len=16, key_len=16):
        self.msg_len = msg_len
        self.key_len = key_len
        self.alice = AliceNet(msg_len, key_len)
        self.bob = BobNet(msg_len, key_len)
        self.eve = EveNet(msg_len)

    def train(self, epochs=10000, batch_size=256):
        optimizer_alice = optim.Adam(self.alice.parameters(), lr=0.0008)
        optimizer_bob = optim.Adam(self.bob.parameters(), lr=0.0008)
        optimizer_eve = optim.Adam(self.eve.parameters(), lr=0.0008)

        # 引入学习率调度器：每 2500 轮降低学习率，提升精细度
        scheduler_a = optim.lr_scheduler.StepLR(optimizer_alice, step_size=2500, gamma=0.5)
        scheduler_b = optim.lr_scheduler.StepLR(optimizer_bob, step_size=2500, gamma=0.5)

        criterion = nn.MSELoss()

        print(f"[GAN] 启动深度优化训练 | 目标: 0 误码还原")
        
        for epoch in range(epochs):
            # 准备随机数据
            p = torch.randint(0, 2, (batch_size, self.msg_len)).float() * 2 - 1
            k = torch.randn(batch_size, self.key_len)

            # --- 训练 Alice & Bob (目标: 还原 + 防御) ---
            optimizer_alice.zero_grad()
            optimizer_bob.zero_grad()

            c = self.alice(p, k)
            p_bob = self.bob(c, k)
            p_eve = self.eve(c)

            loss_bob = criterion(p_bob, p)
            loss_eve = criterion(p_eve, p)
            
            # 优化点 1: 提升 Bob 的损失权重 (从 1.0 提升到 2.0)
            # 优化点 2: 增加防御项的非线性强度
            loss_alice = 2.0 * loss_bob + (1.0 - loss_eve)**2

            loss_alice.backward()
            optimizer_alice.step()
            optimizer_bob.step()

            # --- 训练 Eve (目标: 破解) ---
            optimizer_eve.zero_grad()
            # 重新生成密文防止过度拟合
            c_for_eve = self.alice(p, k).detach()
            p_eve_actual = self.eve(c_for_eve)
            loss_eve_actual = criterion(p_eve_actual, p)
            
            loss_eve_actual.backward()
            optimizer_eve.step()
            
            # 调度器推进
            scheduler_a.step()
            scheduler_b.step()

            if epoch % 1000 == 0:
                print(f"Epoch {epoch:4d} | Bob L1: {loss_bob.item():.6f} | Eve L1: {loss_eve_actual.item():.4f} | LR: {optimizer_alice.param_groups[0]['lr']:.6f}")

        # 训练结束保存
        torch.save({
            'alice_state_dict': self.alice.state_dict(),
            'bob_state_dict': self.bob.state_dict(),
            'eve_state_dict': self.eve.state_dict()
        }, "neural_weights.pth")
        print(f"[Engine] 优化模型已保存。")

    def encrypt(self, msg_bits, shared_key):
        self.alice.eval()
        p = torch.tensor(msg_bits).float().view(1, -1) * 2 - 1
        k = torch.tensor(shared_key).float().view(1, -1)
        with torch.no_grad():
            return self.alice(p, k).squeeze().numpy()

    def decrypt(self, cipher_vec, shared_key):
        self.bob.eval()
        c = torch.tensor(cipher_vec).float().view(1, -1)
        k = torch.tensor(shared_key).float().view(1, -1)
        with torch.no_grad():
            p_rec = self.bob(c, k)
        return (p_rec.squeeze().numpy() > 0).astype(int)

if __name__ == "__main__":
    msg_size = 16
    engine = GANCryptoEngine(msg_len=msg_size)
    
    # 增加到 10000 轮以配合 LR 调度
    engine.train(epochs=10000)
    
    # 审计
    print("\n" + "="*20 + " 100组随机样本审计 " + "="*20)
    bob_errors = 0
    for _ in range(100):
        m = np.random.randint(0, 2, msg_size)
        k = np.random.randn(msg_size)
        c = engine.encrypt(m, k)
        r = engine.decrypt(c, k)
        bob_errors += np.sum(m != r)
    
    print(f"审计完成！Bob 总计位错误数: {bob_errors}")
    print(f"平均误码率: {(bob_errors/(100*msg_size))*100:.2f}%")