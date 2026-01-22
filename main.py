import numpy as np
import torch
from gan_engine import GANCryptoEngine
from crypto_utils import CryptoUtils
# 修正导入：将 NeuralBlockchain 改为 Blockchain
from blockchain_core import Blockchain, Transaction

def run_demo():
    print("=== 神经加密区块链集成演示 ===\n")

    # 1. 初始化并训练 GAN 引擎 (模块 1)
    msg_len = 16
    engine = GANCryptoEngine(msg_len=msg_len)
    # 增加训练轮数以提高解密准确度
    print("[系统] 正在进行深度对抗训练，请稍候...")
    engine.train(epochs=5000) 

    # 2. 初始化区块链 (模块 3)
    blockchain = Blockchain()

    # 3. 模拟用户 Alice 和 Bob 的身份密钥 (模块 2)
    alice_sk, alice_pk = CryptoUtils.generate_key_pair()
    
    # 模拟 GAN 通信所需的共享对称 Key
    gan_shared_key = np.random.randn(msg_len)

    # 4. 模拟交易生命周期
    # A. 准备原始数据
    raw_data = [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
    print(f"原始明文消息: {raw_data}")

    # B. 加密阶段 (GAN Engine)
    ciphertext = engine.encrypt(raw_data, gan_shared_key)
    print(f"GAN 加密完成，生成密文向量 (长度 {len(ciphertext)})")

    # C. 准备交易元数据
    fake_cid = "QmXoyp_GAN_DATA"
    cipher_str = str(ciphertext.tolist())
    
    # 签名阶段
    receiver_id = "Bob_Node"
    data_to_sign = f"{fake_cid}{cipher_str}{receiver_id}"
    signature = CryptoUtils.sign_data(alice_sk, data_to_sign)

    # D. 广播与验证 (Blockchain Core)
    new_tx = Transaction(alice_pk, receiver_id, fake_cid, cipher_str, signature)
    
    if blockchain.add_transaction(new_tx):
        print("交易验证通过（身份签名合法），已加入待处理池。")
    
    # E. 打包入链
    print("正在挖矿打包...")
    if blockchain.mine():
        mined_block = blockchain.chain[-1]
        print(f"新区块已记录! Hash: {mined_block.hash}")

    # F. 接收方 Bob 审计与解密
    print("\n>>> Bob 节点执行审计解密...")
    tx_on_chain = blockchain.chain[-1].transactions[0]
    
    # 从字典中取回密文
    payload = eval(tx_on_chain['hash']) 
    recovered_data = engine.decrypt(payload, gan_shared_key)
    
    recovered_list = recovered_data.tolist()
    print(f"Bob 还原的明文: {recovered_list}")

    # G. 安全性对比：Eve 尝试破解
    c_tensor = torch.tensor(payload).float().view(1, -1)
    with torch.no_grad():
        # Eve 只能接触到密文
        eve_guess = (engine.eve(c_tensor).squeeze().numpy() > 0.5).astype(int)
    print(f"攻击者 Eve 的猜测: {eve_guess.tolist()}")

    # 最终结果判断
    success = np.array_equal(raw_data, recovered_list)
    if success:
        print("\n[结论] 混合系统运行完美：身份验证与神经解密全部成功！")
    else:
        print("\n[警告] 解密数据不匹配。原因：GAN 训练尚未完全收敛或消息噪声过大。")
        print(f"错误位统计: {np.sum(np.array(raw_data) != np.array(recovered_list))} / {msg_len}")

if __name__ == "__main__":
    run_demo()