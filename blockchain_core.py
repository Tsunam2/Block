import json
import time
from crypto_utils import CryptoUtils

class Transaction:
    """
    交易类：封装存证信息。
    包括发送方公钥、接收方ID、IPFS地址(CID)、密文哈希以及数字签名。
    """
    def __init__(self, sender_public_key, receiver_id, ipfs_cid, body_hash, signature):
        # 核心逻辑：如果是RSA公钥对象，序列化为PEM字符串存储；如果是字符串则直接存储
        if not isinstance(sender_public_key, str) and sender_public_key is not None:
            self.sender_pk = CryptoUtils.serialize_public_key(sender_public_key)
        else:
            self.sender_pk = sender_public_key
            
        self.receiver_id = receiver_id
        self.ipfs_cid = ipfs_cid
        self.body_hash = body_hash  # 密文的哈希值
        self.signature = signature
        self.timestamp = time.time()

    def to_dict(self):
        """
        将交易转换为字典。
        键名必须与 main.py 中的读取逻辑严格一致。
        """
        return {
            "sender_pk": self.sender_pk,
            "receiver_id": self.receiver_id,
            "cid": self.ipfs_cid,          # 对应 main.py 中的 on_chain_tx['cid']
            "content_hash": self.body_hash, # 对应 main.py 中的 on_chain_tx['content_hash']
            "signature": self.signature,
            "timestamp": self.timestamp
        }

class Block:
    """
    区块类：包含多个交易的集合。
    """
    def __init__(self, index, transactions, previous_hash):
        self.index = index
        self.timestamp = time.time()
        # 确保存储的是交易字典列表
        self.transactions = [tx.to_dict() if hasattr(tx, 'to_dict') else tx for tx in transactions]
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "transactions": self.transactions,
            "previous_hash": self.previous_hash
        }, sort_keys=True)
        return CryptoUtils.get_sha256(block_string)

class Blockchain:
    """
    区块链管理器。
    """
    def __init__(self):
        self.chain = [self.create_genesis_block()]
        self.pending_transactions = []

    def create_genesis_block(self):
        """生成创世区块"""
        return Block(0, [], "0")

    def get_last_block(self):
        """
        获取链上最后一个区块。
        修复 AttributeError: 'Blockchain' object has no attribute 'get_last_block'
        """
        return self.chain[-1]

    def add_transaction(self, transaction):
        """
        验证并添加交易。
        """
        try:
            # 1. 还原公钥对象用于验签
            pub_key_obj = CryptoUtils.deserialize_public_key(transaction.sender_pk)
            
            # 2. 验证签名 (注意：签名校验的数据必须与 Alice 签名时拼凑的字符串完全一致)
            # 如果 main.py 报错，请检查 CryptoUtils.verify_signature 的参数顺序
            is_valid = CryptoUtils.verify_signature(
                pub_key_obj, 
                transaction.signature, 
                transaction.body_hash
            )
            
            if is_valid:
                self.pending_transactions.append(transaction)
                pk_brief = transaction.sender_pk[:25].replace('\n', '')
                print(f"[Blockchain] 交易验证通过，已入池。发送方: {pk_brief}...")
                return True
            else:
                print("[Blockchain] 警告：签名验证失败！")
                return False
        except Exception as e:
            print(f"[Blockchain] 交易处理异常: {e}")
            return False

    def mine(self):
        """
        打包待处理交易。
        """
        if not self.pending_transactions:
            return False
        
        new_block = Block(
            len(self.chain), 
            self.pending_transactions, 
            self.get_last_block().hash
        )
        self.chain.append(new_block)
        self.pending_transactions = []
        print(f"[Blockchain] 区块 #{new_block.index} 已成功挂载至主链。")
        return True