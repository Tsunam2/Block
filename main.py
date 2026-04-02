import numpy as np
import torch
import json
import os
from gan_engine import GANCryptoEngine
from crypto_utils import CryptoUtils
from blockchain_core import Blockchain, Transaction

# 模拟 IPFS 层
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

    msg_len = 16
    engine = GANCryptoEngine(msg_len=msg_len)
    
    # --- 修改点 1: 优先加载最佳模型 (best_checkpoint.pth) ---
    weights_path = "best_checkpoint.pth" 
    if not os.path.exists(weights_path):
        weights_path = "checkpoint.pth" # 退而求其次加载普通存档

    if os.path.exists(weights_path):
        print(f"[系统] 正在加载性能最优模型: {weights_path}...")
        # 增加 weights_only=False 以兼容旧版保存的 stats 字典
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
        engine.alice.load_state_dict(checkpoint['alice'])
        engine.bob.load_state_dict(checkpoint['bob'])
        engine.eve.load_state_dict(checkpoint['eve'])
        engine.alice.eval()
        engine.bob.eval()
    else:
        print("[系统] 未发现权重文件，开始 15000 轮完整训练...")
        engine.train(epochs=15001)
        # 训练完后 engine 会自动生成 best_checkpoint.pth

    ipfs = MockIPFS()
    blockchain = Blockchain()
    
    # 2. 生成身份密钥 (RSA) 和 神经加密共享密钥
    alice_sk, alice_pk = CryptoUtils.generate_key_pair()
    # --- 修改点 2: 显式指定密钥生成方式，确保与训练分布一致 ---
    shared_neural_key = np.random.normal(0, 1, msg_len).astype(np.float32)

    # --- 流程 A: Alice ---
    print("\n--- 流程 A: Alice 执行加密与上链 ---")
    raw_bits = [1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0]
    print(f"[Alice] 原始比特流: {raw_bits}")

    # 加密
    cipher_vec = engine.encrypt(raw_bits, shared_neural_key)
    cipher_str = json.dumps(cipher_vec.tolist())
    
    cid = ipfs.upload(cipher_str)
    h_value = CryptoUtils.get_sha256(cipher_str)
    signature = CryptoUtils.sign_data(alice_sk, h_value)
    tx = Transaction(alice_pk, "Bob_Node_ID", cid, h_value, signature)
    
    # --- 流程 B: 验证 ---
    print("\n--- 流程 B: 区块链节点验证 ---")
    if blockchain.add_transaction(tx):
        blockchain.mine()
    else:
        print("[错误] 签名验证失败！")
        return

    # --- 流程 C: Bob ---
    print("\n--- 流程 C: Bob 提取数据与审计 ---")
    last_block = blockchain.get_last_block()
    if not last_block.transactions:
        print("[错误] 区块链中无交易数据")
        return
        
    on_chain_tx = last_block.transactions[0]
    sender_pk_obj = CryptoUtils.deserialize_public_key(on_chain_tx['sender_pk'])
    
    if CryptoUtils.verify_signature(sender_pk_obj, on_chain_tx['signature'], on_chain_tx['content_hash']):
        print("[Bob] 1. RSA 身份验签通过。")
        downloaded_cipher_str = ipfs.download(on_chain_tx['cid'])
        
        if CryptoUtils.get_sha256(downloaded_cipher_str) == on_chain_tx['content_hash']:
            print("[Bob] 2. 密文完整性校验通过。")
            
            # 解密
            cipher_vec_final = np.array(json.loads(downloaded_cipher_str))
            recovered_bits = engine.decrypt(cipher_vec_final, shared_neural_key)
            
            print(f"\n[结果] Bob 还原的明文: {recovered_bits.tolist()}")
            if np.array_equal(raw_bits, recovered_bits):
                print(">>> 演示成功：全链路验证一致！ (准确率 100%) <<<")
            else:
                diff = np.sum(np.array(raw_bits) != recovered_bits)
                error_rate = (diff / msg_len) * 100
                print(f">>> 演示失败：解密结果有 {diff} 位错误 (错误率 {error_rate:.2%})。 <<<")

if __name__ == "__main__":
    run_integrated_demo()