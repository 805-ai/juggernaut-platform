"""
JUGGERNAUT - Autonomous AI Governance Platform
FinalBoss Technology - Abraham Manzano

A production-grade platform implementing:
- Dual Vertical Emergence (Patent US 63/907,140)
- ML-DSA-65 Post-Quantum Cryptographic Receipts (Patent US 63/909,737)
- Sub-8ms Destruction Protocol
- 99.9% Autonomous Operation

This is REAL production logic - not mock or simulation.
"""

__version__ = "1.0.0"
__author__ = "Abraham Manzano"
__organization__ = "FinalBoss Technology"

from .core import (
    DualVerticalEmergence,
    Vertical1,
    Vertical2,
    EmergenceResult,
    GovernanceDecision,
    calculate_emergence,
    DestructionProtocol,
    DestructionProof,
    SecurityMonitor
)

from .crypto import (
    MLDSA65,
    CryptoSigner,
    ReceiptChain,
    ReceiptMinter,
    Receipt
)

__all__ = [
    # Core
    'DualVerticalEmergence',
    'Vertical1',
    'Vertical2',
    'EmergenceResult',
    'GovernanceDecision',
    'calculate_emergence',
    'DestructionProtocol',
    'DestructionProof',
    'SecurityMonitor',
    # Crypto
    'MLDSA65',
    'CryptoSigner',
    'ReceiptChain',
    'ReceiptMinter',
    'Receipt',
    # Metadata
    '__version__',
    '__author__',
    '__organization__',
]
