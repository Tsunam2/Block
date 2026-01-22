import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# =================================================================
# 核心架构：密文空间扩展 (16位 -> 32位)
# =================================================================

class CryptoBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim), # 改用 LayerNorm 增强稳定性
            nn.GELU(),             # 使用 GELU 替代 LeakyReLU
            nn.Dropout(0.05)
        )
    def forward(self, x):
        return self.net(x)

class AliceNet(nn.Module):
    def __init__(self, msg_len, key_len, cipher_len):
        super().__init__()
        self.main = nn.Sequential(
            CryptoBlock(msg_len + key_len, 512),
            CryptoBlock(512, 512),
            nn.Linear(512, cipher_len), # 输出 32 维
            nn.Tanh()
        )
    def forward(self, p, pk):
        return self.main(torch.cat((p, pk), dim=1))

class BobNet(nn.Module):
    def __init__(self, cipher_len, key_len, msg_len):
        super().__init__()
        self.main = nn.Sequential(
            CryptoBlock(cipher_len + key_len, 512),
            CryptoBlock(512, 512),
            nn.Linear(512, msg_len),
            nn.Tanh()
        )
    def forward(self, c, sk):
        return self.main(torch.cat((c, sk), dim=1))

class EveNet(nn.Module):
    def __init__(self, cipher_len, msg_len):
        super().__init__()
        # 赋予 Eve 更强的计算能力以进行严格测试
        self.main = nn.Sequential(
            CryptoBlock(cipher_len, 1024),
            CryptoBlock(1024, 1024),
            CryptoBlock(1024, 512),
            nn.Linear(512, msg_len),
            nn.Tanh()
        )
    def forward(self, c):
        return self.main(c)

# =================================================================
# 训练引擎
# =================================================================

class UltimateCryptoEngine:
    def __init__(self, msg_len=16, key_len=16, cipher_len=32):
        self.msg_len = msg_len
        self.key_len = key_len
        self.cipher_len = cipher_len
        
        self.alice = AliceNet(msg_len, key_len, cipher_len)
        self.bob = BobNet(cipher_len, key_len, msg_len)
        self.eve = EveNet(cipher_len, msg_len)
        
        self.criterion = nn.L1Loss()
        self.opt_ab = optim.AdamW(list(self.alice.parameters()) + list(self.bob.parameters()), lr=0.0003)
        self.opt_e = optim.AdamW(self.eve.parameters(), lr=0.0003)
        
        self.sched_ab = optim.lr_scheduler.CosineAnnealingLR(self.opt_ab, T_max=10000)

    def train(self, epochs=10000, batch_size=512):
        print(f"[GAN] 终极防御训练启动 | 密文扩展: {self.cipher_len}D")
        
        for epoch in range(epochs):
            p = torch.randint(0, 2, (batch_size, self.msg_len)).float() * 2 - 1
            pk = torch.randn(batch_size, self.key_len)
            
            # --- 强化 Eve 训练 ---
            for _ in range(2):
                self.opt_e.zero_grad()
                with torch.no_grad():
                    c = self.alice(p, pk)
                p_eve = self.eve(c)
                l_eve = self.criterion(p_eve, p)
                l_eve.backward()
                self.opt_e.step()

            # --- 训练 Alice & Bob ---
            self.opt_ab.zero_grad()
            c_new = self.alice(p, pk)
            
            # 加入一点点扰动，确保护盾坚固
            noise = torch.randn_like(c_new) * 0.005
            p_bob = self.bob(c_new + noise, pk)
            p_eve_adv = self.eve(c_new)
            
            l_rec = self.criterion(p_bob, p)
            l_eve_adv = self.criterion(p_eve_adv, p)
            
            # 改进的惩罚函数：将 Eve 强行推向 L1=1.0 (即完全猜错或盲猜)
            penalty = 4.0 * torch.pow(torch.clamp(0.9 - l_eve_adv, min=0), 2)
            
            total_loss = l_rec + penalty
            total_loss.backward()
            self.opt_ab.step()
            self.sched_ab.step()

            if epoch % 1000 == 0:
                print(f"Epoch {epoch:5d} | Bob L1: {l_rec.item():.5f} | Eve L1: {l_eve.item():.4f} | Pen: {penalty.item():.4f}")

    def audit(self, test_rounds=500):
        self.alice.eval(); self.bob.eval(); self.eve.eval()
        b_errs, e_hits = 0, 0
        
        with torch.no_grad():
            for _ in range(test_rounds):
                m = torch.randint(0, 2, (1, self.msg_len)).float() * 2 - 1
                k = torch.randn(1, self.key_len)
                
                c = self.alice(m, k)
                r = self.bob(c, k)
                e_out = self.eve(c)
                
                m_bits = (m.squeeze() > 0).int()
                r_bits = (r.squeeze() > 0).int()
                e_bits = (e_out.squeeze() > 0).int()
                
                b_errs += torch.sum(m_bits != r_bits).item()
                e_hits += torch.sum(m_bits == e_bits).item()

        total_bits = test_rounds * self.msg_len
        print("\n" + "="*40)
        print(f"最终审计报告 (测试总比特: {total_bits})")
        print(f"Bob 准确率: {1 - b_errs/total_bits:.4%}")
        print(f"Eve 命中率: {e_hits/total_bits:.2%}")
        print("="*40)

if __name__ == "__main__":
    engine = UltimateCryptoEngine(cipher_len=32) # 关键：扩展密文空间
    engine.train(epochs=10000)
    engine.audit()