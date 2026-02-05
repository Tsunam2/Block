import numpy as np
import torch
import json
import os
from gan_engine import GANCryptoEngine
from crypto_utils import CryptoUtils
from blockchain_core import Blockchain, Transaction

# 模拟 IPFS 去中心化存储层
class MockIPFS:
    def __init__(self):
        self.network_storage = {}
    
    def upload(self, data):
        cid = f"QmNeural{CryptoUtils.get_sha256(data)[:16]}"
        self.network_storage[cid] = data
        return cid
    
    def download(self, cid):
        return self.network_storage.get(cid)

def run_integrated_demo():
    print(">>> 启动集成系统测试：神经加密 + 区块链存证 <<<\n")

    # 1. 系统初始化
    msg_len = 16
    engine = GANCryptoEngine(msg_len=msg_len)
    
    # 核心检查：如果存在预训练权重，直接加载
    # 修正：gan_engine.py 默认保存名为 checkpoint.pth
    weights_path = "checkpoint.pth" 
    if os.path.exists(weights_path):
        print(f"[系统] 检测到 {weights_path}，正在加载预训练神经模型...")
        # 修正：加载逻辑对齐 gan_engine 的 state 结构
        checkpoint = torch.load(weights_path, weights_only=True)
        engine.alice.load_state_dict(checkpoint['alice'])
        engine.bob.load_state_dict(checkpoint['bob'])
        engine.alice.eval()
        engine.bob.eval()
    else:
        print("[系统] 未发现权重文件，正在现场训练 GAN 引擎 (2000 epochs)...")
        engine.train(epochs=2000) 

    ipfs = MockIPFS()
    blockchain = Blockchain()
    
    # 2. 生成身份密钥 (RSA)
    alice_sk, alice_pk = CryptoUtils.generate_key_pair()
    # 修正：确保 key 向量是 float32 类型，防止 torch 计算报错
    shared_neural_key = np.random.randn(msg_len).astype(np.float32)

    # ==========================================
    # 流程 A: 发送方 Alice 操作
    # ==========================================
    print("\n--- 流程 A: Alice 执行加密与上链 ---")
    raw_bits = [1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0]
    print(f"[Alice] 原始明文: {raw_bits}")

    # Step 1: 神经加密 (使用 engine 新接口)
    cipher_vec = engine.encrypt(raw_bits, shared_neural_key)
    cipher_str = json.dumps(cipher_vec.tolist()) 
    
    # Step 2: 上传数据到 IPFS
    cid = ipfs.upload(cipher_str)
    
    # Step 3: 计算密文哈希 H 并用 RSA 私钥签名
    h_value = CryptoUtils.get_sha256(cipher_str)
    signature = CryptoUtils.sign_data(alice_sk, h_value)
    
    # Step 4: 构造交易并提交
    tx = Transaction(alice_pk, "Bob_Node_ID", cid, h_value, signature)
    
    # ==========================================
    # 流程 B: 区块链网络验证
    # ==========================================
    print("\n--- 流程 B: 区块链节点验证 ---")
    if blockchain.add_transaction(tx):
        blockchain.mine()
    else:
        print("[错误] 签名验证失败！")
        return

    # ==========================================
    # 流程 C: 接收方 Bob 操作
    # ==========================================
    print("\n--- 流程 C: Bob 提取数据与审计 ---")
    
    on_chain_tx = blockchain.get_last_block().transactions[0]
    sender_pk_obj = CryptoUtils.deserialize_public_key(on_chain_tx['sender_pk'])
    
    if CryptoUtils.verify_signature(sender_pk_obj, on_chain_tx['signature'], on_chain_tx['content_hash']):
        print("[Bob] 1. RSA 身份验签通过。")
        
        downloaded_cipher_str = ipfs.download(on_chain_tx['cid'])
        if CryptoUtils.get_sha256(downloaded_cipher_str) == on_chain_tx['content_hash']:
            print("[Bob] 2. 密文完整性校验通过。")
            
            # 神经解密 (使用 engine 新接口)
            cipher_vec_final = np.array(json.loads(downloaded_cipher_str))
            recovered_bits = engine.decrypt(cipher_vec_final, shared_neural_key)
            
            print(f"\n[结果] Bob 还原的明文: {recovered_bits.tolist()}")
            if np.array_equal(raw_bits, recovered_bits):
                print(">>> 演示成功：全链路验证一致！ <<<")
            else:
                diff = np.sum(np.array(raw_bits) != recovered_bits)
                print(f">>> 演示失败：解密结果有 {diff} 位错误。 <<<")

if __name__ == "__main__":
    run_integrated_demo()