import hashlib
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization

class CryptoUtils:
    """
    提供系统中传统的密码学支持：数字签名、验签和哈希转换。
    """
    
    @staticmethod
    def generate_key_pair():
        """
        生成一对 RSA 密钥，用于数字签名（这是身份证明，非神经网络权重）。
        """
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        public_key = private_key.public_key()
        return private_key, public_key

    @staticmethod
    def sign_data(private_key, data_string):
        """
        使用私钥对数据字符串进行签名。返回 16 进制字符串。
        """
        signature = private_key.sign(
            data_string.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return signature.hex()

    @staticmethod
    def verify_signature(public_key, signature_hex, data_string):
        """
        使用公钥验证数字签名的有效性。
        """
        try:
            signature = bytes.fromhex(signature_hex)
            public_key.verify(
                signature,
                data_string.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False

    @staticmethod
    def get_sha256(data_string):
        """计算字符串的 SHA-256 哈希值"""
        return hashlib.sha256(data_string.encode()).hexdigest()

# --- 功能冒烟测试 ---
if __name__ == "__main__":
    print(">>> 启动 CryptoUtils 模块测试 <<<")
    
    # 1. 模拟 Alice 生成签名密钥对
    alice_priv, alice_pub = CryptoUtils.generate_key_pair()
    
    # 2. 模拟要存证的交易载荷（例如：IPFS地址 + 密文哈希）
    tx_payload = "CID:2218d9275e4d09f5;BodyHash:2218d927..."
    
    # 3. 发送方执行签名
    sig = CryptoUtils.sign_data(alice_priv, tx_payload)
    print(f"[签名] 生成成功, 长度: {len(sig)} 字符")
    
    # 4. 验证节点执行验证
    is_valid = CryptoUtils.verify_signature(alice_pub, sig, tx_payload)
    print(f"[验证] 原始数据验证结果: {'通过' if is_valid else '失败'}")
    
    # 5. 篡改测试
    tampered_data = tx_payload + "modified_by_attacker"
    is_valid_tampered = CryptoUtils.verify_signature(alice_pub, sig, tampered_data)
    print(f"[验证] 篡改数据验证结果: {'通过' if is_valid_tampered else '失败'}")