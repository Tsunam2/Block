import hashlib
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization

class CryptoUtils:
    """
    提供系统中传统的密码学支持：
    1. RSA 密钥对生成 (用于身份认证)
    2. 数字签名与验签 (确保交易不可篡改)
    3. 公钥的序列化与反序列化 (用于在区块链/网络中传输)
    4. SHA-256 哈希计算 (用于区块哈希)
    """
    
    @staticmethod
    def generate_key_pair():
        """
        生成一对 RSA 密钥。
        私钥用于签名，公钥作为区块链上的用户地址/身份标识。
        """
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        public_key = private_key.public_key()
        return private_key, public_key

    @staticmethod
    def serialize_public_key(public_key):
        """
        将 RSA 公钥对象转换为 PEM 格式的字节串（再转为字符串），以便存储在区块中。
        """
        pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return pem.decode('utf-8')

    @staticmethod
    def deserialize_public_key(pem_string):
        """
        将 PEM 格式的字符串还原为 RSA 公钥对象。
        """
        return serialization.load_pem_public_key(
            pem_string.encode('utf-8')
        )

    @staticmethod
    def sign_data(private_key, data_string):
        """
        使用私钥对数据字符串进行签名。
        返回结果为 16 进制字符串。
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
        """
        计算字符串的标准 SHA-256 哈希值。
        """
        return hashlib.sha256(data_string.encode()).hexdigest()

# --- 模块测试 ---
if __name__ == "__main__":
    print(">>> 启动 CryptoUtils 模块测试 <<<")
    
    # 1. 测试密钥生成与序列化
    priv, pub = CryptoUtils.generate_key_pair()
    pub_str = CryptoUtils.serialize_public_key(pub)
    print(f"[测试] 公钥序列化成功，长度: {len(pub_str)}")
    
    # 2. 测试反序列化与签名验证
    pub_obj = CryptoUtils.deserialize_public_key(pub_str)
    msg = "test_message"
    sig = CryptoUtils.sign_data(priv, msg)
    if CryptoUtils.verify_signature(pub_obj, sig, msg):
        print("[测试] 序列化后的公钥验签成功！")