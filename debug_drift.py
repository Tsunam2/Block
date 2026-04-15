import torch
import os
from gan_engine import AliceNet

# 配置与加载
MSG_LEN, KEY_LEN = 16, 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alice = AliceNet(MSG_LEN, KEY_LEN).to(DEVICE)
checkpoint = torch.load("best_checkpoint.pth", map_location=DEVICE, weights_only=False)
alice.load_state_dict(checkpoint['alice'] if 'alice' in checkpoint else checkpoint)
alice.eval()

# 准备数据
m1 = torch.randint(0, 2, (1, MSG_LEN)).float().to(DEVICE) * 2 - 1
m2 = m1.clone()
m2[0, 0] = -m2[0, 0] # 翻转第 0 位
k = torch.randn(1, KEY_LEN).to(DEVICE)

with torch.no_grad():
    c1 = alice(m1, k)
    c2 = alice(m2, k)
    
    print(f"密文 1 前 5 位: {c1[0, :5].cpu().numpy()}")
    print(f"密文 2 前 5 位: {c2[0, :5].cpu().numpy()}")
    
    l1_dist = torch.mean(torch.abs(c1 - c2)).item()
    print(f"\n平均数值漂移 (L1 Drift): {l1_dist:.8f}")
    
    flip_count = torch.sum((c1 > 0) != (c2 > 0)).item()
    print(f"比特翻转个数: {flip_count} / {MSG_LEN}")