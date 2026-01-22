import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
#v1.0
# --- 1. 神经网络架构设计 ---

class AliceNet(nn.Module):
    """Alice: 加密网络 (Sender)"""
    def __init__(self, msg_len, key_len):
        super(AliceNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(msg_len + key_len, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, msg_len),
            nn.Tanh() 
        )

    def forward(self, p, pk):
        x = torch.cat((p, pk), dim=1)
        return self.fc(x)

class BobNet(nn.Module):
    """Bob: 解密网络 (Receiver)"""
    def __init__(self, msg_len, key_len):
        super(BobNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(msg_len + key_len, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, msg_len),
            nn.Tanh()
        )

    def forward(self, c, sk):
        x = torch.cat((c, sk), dim=1)
        return self.fc(x)

class EveNet(nn.Module):
    """Eve: 攻击网络 (Attacker)"""
    def __init__(self, msg_len):
        super(EveNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(msg_len, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, msg_len),
            nn.Tanh()
        )

    def forward(self, c):
        return self.fc(c)

# --- 2. 增强型训练与加密引擎 ---

class GANCryptoEngine:
    def __init__(self, msg_len=16, key_len=16):
        self.msg_len = msg_len
        self.key_len = key_len
        self.alice = AliceNet(msg_len, key_len)
        self.bob = BobNet(msg_len, key_len)
        self.eve = EveNet(msg_len)
        
        self.criterion = nn.MSELoss()
        self.opt_alice_bob = optim.Adam(
            list(self.alice.parameters()) + list(self.bob.parameters()), 
            lr=0.0004
        )
        self.opt_eve = optim.Adam(self.eve.parameters(), lr=0.0004)
        
        self.history = {"bob_mse": [], "eve_mse": []}

    def train(self, epochs=3000, batch_size=128):
        """执行增强对抗训练并记录历史"""
        print(f"[GAN] 开始对抗训练 (消息长度: {self.msg_len}, 轮数: {epochs})...")
        
        for epoch in range(epochs):
            # 准备训练数据
            p = torch.randint(0, 2, (batch_size, self.msg_len)).float() * 2 - 1
            pk = torch.randn(batch_size, self.key_len)
            sk = pk 

            # --- 优化 Eve ---
            for _ in range(2):
                self.opt_eve.zero_grad()
                with torch.no_grad():
                    c = self.alice(p, pk)
                p_eve = self.eve(c)
                l_eve = self.criterion(p_eve, p)
                l_eve.backward()
                self.opt_eve.step()

            # --- 优化 Alice & Bob ---
            self.opt_alice_bob.zero_grad()
            c_new = self.alice(p, pk)
            p_bob = self.bob(c_new, sk)
            
            p_eve_adv = self.eve(c_new)
            l_eve_adv = self.criterion(p_eve_adv, p)
            l_rec = self.criterion(p_bob, p)
            
            # 对抗损失函数: 惩罚 Eve 预测得太准
            l_adv = l_rec + torch.pow(1.0 - l_eve_adv, 2)
            
            l_adv.backward()
            self.opt_alice_bob.step()

            # 记录历史
            if epoch % 100 == 0:
                self.history["bob_mse"].append(l_rec.item())
                self.history["eve_mse"].append(l_eve.item())

            if epoch % 500 == 0:
                print(f"Epoch {epoch:4d} | Bob MSE: {l_rec.item():.4f} | Eve MSE: {l_eve.item():.4f}")
        
        print("[GAN] 训练完成。\n")

    def save_models(self, path_prefix="crypto_"):
        """保存模型权重"""
        torch.save(self.alice.state_dict(), f"{path_prefix}alice.pth")
        torch.save(self.bob.state_dict(), f"{path_prefix}bob.pth")
        print(f"[系统] 模型权重已保存至 {path_prefix}*.pth")

    def load_models(self, path_prefix="crypto_"):
        """加载模型权重"""
        self.alice.load_state_dict(torch.load(f"{path_prefix}alice.pth"))
        self.bob.load_state_dict(torch.load(f"{path_prefix}bob.pth"))
        self.alice.eval()
        self.bob.eval()
        print(f"[系统] 模型权重已加载。")

    def encrypt(self, msg_bits, pub_key):
        """
        工具方法: 将二进制比特数组加密为密文向量
        msg_bits: list of 0/1, len = msg_len
        """
        p = torch.tensor(msg_bits).float().view(1, -1) * 2 - 1
        pk = torch.tensor(pub_key).float().view(1, -1)
        self.alice.eval()
        with torch.no_grad():
            c = self.alice(p, pk)
        return c.squeeze().numpy()

    def decrypt(self, cipher_vec, priv_key):
        """
        工具方法: 将密文向量还原为二进制比特
        """
        c = torch.tensor(cipher_vec).float().view(1, -1)
        sk = torch.tensor(priv_key).float().view(1, -1)
        self.bob.eval()
        with torch.no_grad():
            p_rec = self.bob(c, sk)
        # 将结果映射回 0/1
        return (p_rec.squeeze().numpy() > 0).astype(int)

if __name__ == "__main__":
    print(">>> 启动生产环境级 GAN 加密引擎测试 <<<")
    msg_size = 16
    engine = GANCryptoEngine(msg_len=msg_size)
    engine.train(epochs=3000)
    
    # 测试真实业务流
    raw_bits = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1]
    shared_key = np.random.randn(msg_size)
    
    print(f"原始明文比特: {raw_bits}")
    
    # 执行加密
    ciphertext = engine.encrypt(raw_bits, shared_key)
    print(f"生成的密文 (前5位): {ciphertext[:5]}...")
    
    # 执行解密
    recovered_bits = engine.decrypt(ciphertext, shared_key)
    print(f"Bob 还原比特: {recovered_bits.tolist()}")
    
    # 模拟攻击者
    c_tensor = torch.tensor(ciphertext).float().view(1, -1)
    with torch.no_grad():
        eve_guess = (engine.eve(c_tensor).squeeze().numpy() > 0).astype(int)
    print(f"Eve 猜测比特: {eve_guess.tolist()}")

    # 验证
    if np.array_equal(raw_bits, recovered_bits):
        print("\n[验证成功] Bob 已实现 100% 比特还原。")
        match_count = np.sum(np.array(raw_bits) == eve_guess)
        print(f"[安全分析] Eve 猜对比特数: {match_count}/{msg_size} (随机期望: {msg_size/2})")
    
    # 保存结果供区块链模块调用
    # engine.save_models()