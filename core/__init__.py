"""
JUGGERNAUT Core Module
FinalBoss Technology
"""

from .emergence import (
    DualVerticalEmergence,
    Vertical1,
    Vertical2,
    EmergenceResult,
    GovernanceDecision,
    calculate_emergence
)

from .destruction import (
    DestructionProtocol,
    DestructionProof,
    DestructionStatus,
    SecureDataContainer
)

from .security import (
    SecurityMonitor,
    InputValidator,
    RateLimiter,
    SecurityEvent,
    AttackType,
    SecurityAction,
    get_security_monitor
)

__all__ = [
    'DualVerticalEmergence',
    'Vertical1',
    'Vertical2',
    'EmergenceResult',
    'GovernanceDecision',
    'calculate_emergence',
    'DestructionProtocol',
    'DestructionProof',
    'DestructionStatus',
    'SecureDataContainer',
    'SecurityMonitor',
    'InputValidator',
    'RateLimiter',
    'SecurityEvent',
    'AttackType',
    'SecurityAction',
    'get_security_monitor',
]
