import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

#v1.1
# =================================================================
# 1. 强化版架构设计
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
# 2. 训练引擎
# =================================================================

class GANCryptoEngine:
    def __init__(self, msg_len=16, key_len=16):
        self.msg_len = msg_len
        self.key_len = key_len
        self.alice = AliceNet(msg_len, key_len)
        self.bob = BobNet(msg_len, key_len)
        self.eve = EveNet(msg_len)
        
        self.criterion = nn.L1Loss()
        self.opt_ab = optim.Adam(list(self.alice.parameters()) + list(self.bob.parameters()), lr=0.0008)
        self.opt_e = optim.Adam(self.eve.parameters(), lr=0.0008)
        
        self.sched_ab = optim.lr_scheduler.StepLR(self.opt_ab, step_size=2000, gamma=0.5)
        self.sched_e = optim.lr_scheduler.StepLR(self.opt_e, step_size=2000, gamma=0.5)

    def train(self, epochs=8000, batch_size=256):
        print(f"[GAN] 启动训练 | 维度: 512 | 目标: 100% 精度")
        
        for epoch in range(epochs):
            p = torch.randint(0, 2, (batch_size, self.msg_len)).float() * 2 - 1
            pk = torch.randn(batch_size, self.key_len)
            
            # --- 优化 Eve ---
            self.opt_e.zero_grad()
            with torch.no_grad():
                c = self.alice(p, pk)
            p_eve = self.eve(c)
            l_eve = self.criterion(p_eve, p)
            l_eve.backward()
            self.opt_e.step()

            # --- 优化 Alice & Bob ---
            self.opt_ab.zero_grad()
            c_new = self.alice(p, pk)
            p_bob = self.bob(c_new, pk)
            p_eve_adv = self.eve(c_new)
            
            l_rec = self.criterion(p_bob, p)
            l_eve_adv = self.criterion(p_eve_adv, p)
            
            # 惩罚项: (1.0 - Leve_adv)^2
            expected_error = 1.0
            penalty = torch.pow(torch.clamp(expected_error - l_eve_adv, min=0), 2)
            
            total_loss = l_rec + penalty
            total_loss.backward()
            self.opt_ab.step()
            
            self.sched_ab.step()
            self.sched_e.step()

            if epoch % 1000 == 0:
                print(f"Epoch {epoch:4d} | Bob L1: {l_rec.item():.4f} | Eve L1: {l_eve.item():.4f} | Pen: {penalty.item():.4f}")

    def encrypt(self, msg_bits, pub_key):
        self.alice.eval()
        p = torch.tensor(msg_bits).float().view(1, -1) * 2 - 1
        pk = torch.tensor(pub_key).float().view(1, -1)
        with torch.no_grad():
            return self.alice(p, pk).squeeze().numpy()

    def decrypt(self, cipher_vec, priv_key):
        self.bob.eval()
        c = torch.tensor(cipher_vec).float().view(1, -1)
        sk = torch.tensor(priv_key).float().view(1, -1)
        with torch.no_grad():
            p_rec = self.bob(c, sk)
        return (p_rec.squeeze().numpy() > 0).astype(int)

# =================================================================
# 3. 最终审计逻辑
# =================================================================

if __name__ == "__main__":
    msg_size = 16
    engine = GANCryptoEngine(msg_len=msg_size)
    engine.train(epochs=8000)
    
    print("\n" + "="*20 + " 100组随机样本审计 " + "="*20)
    test_rounds = 100
    bob_errors = 0
    eve_hits = 0
    
    # 核心修复：切换到 eval 模式
    engine.alice.eval()
    engine.bob.eval()
    engine.eve.eval()

    for _ in range(test_rounds):
        m = np.random.randint(0, 2, msg_size)
        k = np.random.randn(msg_size)
        
        c = engine.encrypt(m, k)
        r = engine.decrypt(c, k)
        
        bob_errors += np.sum(m != r)
        
        with torch.no_grad():
            c_t = torch.tensor(c).float().view(1, -1)
            # 处于 eval 模式下的 Eve 不再会因为单条数据报错
            e_out = engine.eve(c_t)
            e_g = (e_out.squeeze().numpy() > 0).astype(int)
        eve_hits += np.sum(m == e_g)

    total_bits = test_rounds * msg_size
    print(f"审计完成！")
    print(f"Bob 平均位错误率: {bob_errors / total_bits:.2%}")
    print(f"Eve 平均位命中率: {eve_hits / total_bits:.2%} (理想为 50.0% 附近)")
    print(f"详细位统计: 错误位 {bob_errors} / {total_bits} | 命中位 {eve_hits} / {total_bits}")