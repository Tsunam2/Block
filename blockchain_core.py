import hashlib
import json
import time
import sys
import os

# 将根目录加入路径以便导入 crypto_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crypto_utils import CryptoUtils

class Transaction:
    def __init__(self, sender_public_key, receiver_id, ipfs_cid, body_hash, signature):
        self.sender_public_key = sender_public_key  # 这是一个 RSA 公钥对象
        self.receiver_id = receiver_id
        self.ipfs_cid = ipfs_cid
        self.body_hash = body_hash
        self.signature = signature
        self.timestamp = time.time()

    def get_data_to_sign(self):
        """
        定义哪些字段参与签名，必须与发送方签名时使用的数据一致
        """
        return f"{self.ipfs_cid}{self.body_hash}{self.receiver_id}"

    def to_dict(self):
        """
        将交易对象转换为字典格式，方便 JSON 序列化
        """
        # 使用 repr() 替代可能的旧式反引号，确保在 Python 3 中完全兼容
        pk_repr = repr(self.sender_public_key)
        return {
            "sender_pk_hash": hashlib.sha256(pk_repr.encode()).hexdigest()[:16],
            "receiver": self.receiver_id,
            "cid": self.ipfs_cid,
            "hash": self.body_hash,
            "sig": self.signature,
            "time": self.timestamp
        }

class Block:
    def __init__(self, index, transactions, previous_hash):
        self.index = index
        self.timestamp = time.time()
        self.transactions = [tx.to_dict() for tx in transactions]
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        block_content = json.dumps({
            "index": self.index,
            "txs": self.transactions,
            "prev_hash": self.previous_hash
        }, sort_keys=True)
        return hashlib.sha256(block_content.encode()).hexdigest()

class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]
        self.pending_transactions = []

    def create_genesis_block(self):
        return Block(0, [], "0")

    def validate_transaction(self, tx):
        """
        验证节点核心逻辑：使用发送方公钥校验签名
        """
        data_to_verify = tx.get_data_to_sign()
        is_valid = CryptoUtils.verify_signature(
            tx.sender_public_key, 
            tx.signature, 
            data_to_verify
        )
        return is_valid

    def add_transaction(self, transaction):
        if self.validate_transaction(transaction):
            self.pending_transactions.append(transaction)
            print("[Blockchain] 验证节点: 签名合法，交易已入池")
            return True
        else:
            print("[Blockchain] 验证节点: 签名非法，交易已被丢弃！")
            return False

    def mine(self):
        if not self.pending_transactions:
            return False
        new_block = Block(len(self.chain), self.pending_transactions, self.chain[-1].hash)
        self.chain.append(new_block)
        self.pending_transactions = []
        print(f"[Blockchain] 区块 #{new_block.index} 挖掘成功")
        return True

# --- 集成测试 ---
if __name__ == "__main__":
    print(">>> 启动集成验证测试 <<<\n")
    
    # 1. 初始化
    my_blockchain = Blockchain()
    alice_priv, alice_pub = CryptoUtils.generate_key_pair()
    
    # 2. 模拟 Alice 发送流程
    fake_cid = "QmXoyp..."
    fake_hash = "sha256_of_ciphertext"
    receiver_info = "Bob_Neural_ID"
    
    # Alice 对数据进行签名
    data_to_sign = f"{fake_cid}{fake_hash}{receiver_info}"
    alice_sig = CryptoUtils.sign_data(alice_priv, data_to_sign)
    
    # 3. 构造交易包并发往区块链
    tx = Transaction(alice_pub, receiver_info, fake_cid, fake_hash, alice_sig)
    
    # 4. 验证并打包
    if my_blockchain.add_transaction(tx):
        my_blockchain.mine()
        print("\n[结果] 交易已成功记录在不可篡改的账本中。")
    
    # 5. 模拟篡改测试
    print("\n[攻击模拟] 攻击者尝试用伪造的签名发送交易...")
    bad_tx = Transaction(alice_pub, receiver_info, fake_cid, fake_hash, "wrong_signature_hex")
    my_blockchain.add_transaction(bad_tx)