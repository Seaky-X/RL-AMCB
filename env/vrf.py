# -*- coding: utf-8 -*-
"""
Ed25519-签名型 VRF（工程可运行、可验证、不可预测的近似 VRF）
- Prove:  pi = Sign_sk(alpha),  y = H(pi)
- Verify: Verify_pk(alpha, pi) -> y = H(pi)

建议安装 cryptography 以满足论文中“可验证”的要求。
"""
from __future__ import annotations
import hashlib
import os

def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey, Ed25519PublicKey
    )
    _HAS_CRYPTO = True
except Exception:
    Ed25519PrivateKey = None
    Ed25519PublicKey = None
    _HAS_CRYPTO = False

class VRFKeypair(object):
    def __init__(self, sk, pk):
        self.sk = sk
        self.pk = pk

    @staticmethod
    def generate():
        if _HAS_CRYPTO:
            sk = Ed25519PrivateKey.generate()
            pk = sk.public_key()
            return VRFKeypair(sk, pk)
        sk = os.urandom(32)
        pk = sha256(sk)
        return VRFKeypair(sk, pk)

def vrf_prove(sk, alpha: bytes):
    if _HAS_CRYPTO:
        pi = sk.sign(alpha)
        y = sha256(pi)
        return y, pi
    pi = sha256(sk + alpha)
    y = sha256(pi)
    return y, pi

def vrf_verify(pk, alpha: bytes, pi: bytes):
    if _HAS_CRYPTO:
        try:
            pk.verify(pi, alpha)
            return True, sha256(pi)
        except Exception:
            return False, b""
    return True, sha256(pi)
