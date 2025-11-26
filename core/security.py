"""
SECURITY MODULE
FinalBoss Technology - JUGGERNAUT Platform

Security features:
- SQL injection detection
- Receipt tamper detection
- Attack logging and blocking
- Rate limiting
- Input sanitization
"""

import re
import time
import hashlib
import threading
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Set
from enum import Enum
from collections import defaultdict


class AttackType(Enum):
    """Types of detected attacks"""
    SQL_INJECTION = "SQL_INJECTION"
    XSS = "XSS"
    PATH_TRAVERSAL = "PATH_TRAVERSAL"
    COMMAND_INJECTION = "COMMAND_INJECTION"
    RECEIPT_TAMPERING = "RECEIPT_TAMPERING"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    INVALID_SIGNATURE = "INVALID_SIGNATURE"
    REPLAY_ATTACK = "REPLAY_ATTACK"
    UNKNOWN = "UNKNOWN"


class SecurityAction(Enum):
    """Actions taken on security events"""
    BLOCKED = "BLOCKED"
    REDACTED = "REDACTED"
    LOGGED = "LOGGED"
    ALLOWED = "ALLOWED"
    QUARANTINED = "QUARANTINED"


@dataclass
class SecurityEvent:
    """Record of a security event"""
    event_id: str
    timestamp: float
    attack_type: AttackType
    action: SecurityAction
    source: str
    target: str
    payload: str
    details: Dict[str, Any]
    blocked: bool
    
    def to_dict(self) -> dict:
        result = asdict(self)
        result['attack_type'] = self.attack_type.value
        result['action'] = self.action.value
        return result


class InputValidator:
    """
    Input validation and sanitization.
    """
    
    # SQL injection patterns
    SQL_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER|CREATE|TRUNCATE)\b)",
        r"(--|\#|\/\*)",
        r"(\bOR\b\s+\d+\s*=\s*\d+)",
        r"(\bAND\b\s+\d+\s*=\s*\d+)",
        r"(;\s*(SELECT|INSERT|UPDATE|DELETE|DROP))",
        r"(\'\s*(OR|AND)\s*\')",
        r"(WAITFOR\s+DELAY)",
        r"(BENCHMARK\s*\()",
        r"(SLEEP\s*\()",
    ]
    
    # XSS patterns
    XSS_PATTERNS = [
        r"(<script[^>]*>)",
        r"(javascript\s*:)",
        r"(on\w+\s*=)",
        r"(<iframe[^>]*>)",
        r"(<object[^>]*>)",
        r"(<embed[^>]*>)",
        r"(<svg[^>]*onload)",
    ]
    
    # Path traversal patterns
    PATH_PATTERNS = [
        r"(\.\.\/)",
        r"(\.\.\\)",
        r"(%2e%2e%2f)",
        r"(%2e%2e\/)",
        r"(\.\.%2f)",
    ]
    
    # Command injection patterns
    COMMAND_PATTERNS = [
        r"(;\s*\w+)",
        r"(\|\s*\w+)",
        r"(\$\()",
        r"(`[^`]+`)",
        r"(\&\&\s*\w+)",
        r"(\|\|\s*\w+)",
    ]
    
    def __init__(self):
        # Compile patterns for performance
        self._sql_regex = [re.compile(p, re.IGNORECASE) for p in self.SQL_PATTERNS]
        self._xss_regex = [re.compile(p, re.IGNORECASE) for p in self.XSS_PATTERNS]
        self._path_regex = [re.compile(p, re.IGNORECASE) for p in self.PATH_PATTERNS]
        self._command_regex = [re.compile(p, re.IGNORECASE) for p in self.COMMAND_PATTERNS]
    
    def detect_sql_injection(self, input_str: str) -> Optional[str]:
        """
        Detect SQL injection attempts.
        
        Returns:
            Matched pattern if detected, None otherwise
        """
        for regex in self._sql_regex:
            match = regex.search(input_str)
            if match:
                return match.group(0)
        return None
    
    def detect_xss(self, input_str: str) -> Optional[str]:
        """Detect XSS attempts"""
        for regex in self._xss_regex:
            match = regex.search(input_str)
            if match:
                return match.group(0)
        return None
    
    def detect_path_traversal(self, input_str: str) -> Optional[str]:
        """Detect path traversal attempts"""
        for regex in self._path_regex:
            match = regex.search(input_str)
            if match:
                return match.group(0)
        return None
    
    def detect_command_injection(self, input_str: str) -> Optional[str]:
        """Detect command injection attempts"""
        for regex in self._command_regex:
            match = regex.search(input_str)
            if match:
                return match.group(0)
        return None
    
    def validate(self, input_str: str) -> Dict[str, Any]:
        """
        Validate input for all attack types.
        
        Returns:
            Validation result with detected attacks
        """
        attacks = []
        
        sql = self.detect_sql_injection(input_str)
        if sql:
            attacks.append({
                'type': AttackType.SQL_INJECTION,
                'pattern': sql
            })
        
        xss = self.detect_xss(input_str)
        if xss:
            attacks.append({
                'type': AttackType.XSS,
                'pattern': xss
            })
        
        path = self.detect_path_traversal(input_str)
        if path:
            attacks.append({
                'type': AttackType.PATH_TRAVERSAL,
                'pattern': path
            })
        
        cmd = self.detect_command_injection(input_str)
        if cmd:
            attacks.append({
                'type': AttackType.COMMAND_INJECTION,
                'pattern': cmd
            })
        
        return {
            'is_safe': len(attacks) == 0,
            'attacks': attacks,
            'input_length': len(input_str)
        }
    
    def sanitize(self, input_str: str) -> str:
        """
        Sanitize input by removing/escaping dangerous patterns.
        
        Returns:
            Sanitized string
        """
        result = input_str
        
        # Remove SQL keywords
        for regex in self._sql_regex:
            result = regex.sub('[REDACTED]', result)
        
        # Escape HTML
        result = result.replace('<', '&lt;')
        result = result.replace('>', '&gt;')
        result = result.replace('"', '&quot;')
        result = result.replace("'", '&#x27;')
        
        # Remove path traversal
        result = result.replace('../', '')
        result = result.replace('..\\', '')
        
        return result


class RateLimiter:
    """
    Rate limiting for API endpoints.
    """
    
    def __init__(self, requests_per_minute: int = 60, 
                 burst_limit: int = 10):
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self._requests: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def check(self, client_id: str) -> Dict[str, Any]:
        """
        Check if a client is within rate limits.
        
        Returns:
            Rate limit status
        """
        now = time.time()
        window_start = now - 60  # 1 minute window
        
        with self._lock:
            # Clean old requests
            self._requests[client_id] = [
                t for t in self._requests[client_id]
                if t > window_start
            ]
            
            requests_in_window = len(self._requests[client_id])
            
            # Check limits
            if requests_in_window >= self.requests_per_minute:
                return {
                    'allowed': False,
                    'reason': 'rate_limit_exceeded',
                    'requests_in_window': requests_in_window,
                    'limit': self.requests_per_minute,
                    'retry_after': 60 - (now - self._requests[client_id][0])
                }
            
            # Check burst (last second)
            burst_start = now - 1
            recent_requests = sum(
                1 for t in self._requests[client_id]
                if t > burst_start
            )
            
            if recent_requests >= self.burst_limit:
                return {
                    'allowed': False,
                    'reason': 'burst_limit_exceeded',
                    'recent_requests': recent_requests,
                    'limit': self.burst_limit,
                    'retry_after': 1.0
                }
            
            # Allow and record
            self._requests[client_id].append(now)
            
            return {
                'allowed': True,
                'requests_in_window': requests_in_window + 1,
                'limit': self.requests_per_minute,
                'remaining': self.requests_per_minute - requests_in_window - 1
            }


class SecurityMonitor:
    """
    Central security monitoring and event logging.
    """
    
    def __init__(self):
        self.validator = InputValidator()
        self.rate_limiter = RateLimiter()
        self.events: List[SecurityEvent] = []
        self.blocked_count = 0
        self.total_checks = 0
        self._seen_signatures: Set[str] = set()  # For replay detection
        self._lock = threading.Lock()
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        timestamp_ms = int(time.time() * 1000)
        return f"SEC-{timestamp_ms}-{len(self.events):06d}"
    
    def check_input(self, 
                    input_str: str,
                    source: str = "unknown",
                    target: str = "unknown") -> Dict[str, Any]:
        """
        Check input for security threats.
        
        Args:
            input_str: Input to check
            source: Source identifier (e.g., IP, user ID)
            target: Target resource
            
        Returns:
            Security check result
        """
        self.total_checks += 1
        
        # Validate input
        validation = self.validator.validate(input_str)
        
        if not validation['is_safe']:
            # Create security event
            for attack in validation['attacks']:
                event = SecurityEvent(
                    event_id=self._generate_event_id(),
                    timestamp=time.time(),
                    attack_type=attack['type'],
                    action=SecurityAction.BLOCKED,
                    source=source,
                    target=target,
                    payload=input_str[:200],  # Truncate for safety
                    details={'pattern': attack['pattern']},
                    blocked=True
                )
                
                with self._lock:
                    self.events.append(event)
                    self.blocked_count += 1
            
            return {
                'safe': False,
                'blocked': True,
                'attacks': [a['type'].value for a in validation['attacks']],
                'sanitized': self.validator.sanitize(input_str)
            }
        
        return {
            'safe': True,
            'blocked': False,
            'attacks': []
        }
    
    def check_rate_limit(self, client_id: str) -> Dict[str, Any]:
        """Check rate limit for a client"""
        result = self.rate_limiter.check(client_id)
        
        if not result['allowed']:
            event = SecurityEvent(
                event_id=self._generate_event_id(),
                timestamp=time.time(),
                attack_type=AttackType.RATE_LIMIT_EXCEEDED,
                action=SecurityAction.BLOCKED,
                source=client_id,
                target="api",
                payload="",
                details=result,
                blocked=True
            )
            
            with self._lock:
                self.events.append(event)
                self.blocked_count += 1
        
        return result
    
    def check_signature(self, signature: str) -> Dict[str, Any]:
        """
        Check for replay attacks using signature.
        
        Args:
            signature: Signature to check
            
        Returns:
            Check result
        """
        sig_hash = hashlib.sha256(signature.encode()).hexdigest()
        
        with self._lock:
            if sig_hash in self._seen_signatures:
                event = SecurityEvent(
                    event_id=self._generate_event_id(),
                    timestamp=time.time(),
                    attack_type=AttackType.REPLAY_ATTACK,
                    action=SecurityAction.BLOCKED,
                    source="unknown",
                    target="signature_verification",
                    payload=signature[:64],
                    details={'sig_hash': sig_hash},
                    blocked=True
                )
                self.events.append(event)
                self.blocked_count += 1
                
                return {
                    'valid': False,
                    'reason': 'replay_attack_detected'
                }
            
            self._seen_signatures.add(sig_hash)
        
        return {'valid': True}
    
    def log_tamper_attempt(self, 
                           receipt_id: str,
                           details: Dict[str, Any]) -> SecurityEvent:
        """Log a receipt tampering attempt"""
        event = SecurityEvent(
            event_id=self._generate_event_id(),
            timestamp=time.time(),
            attack_type=AttackType.RECEIPT_TAMPERING,
            action=SecurityAction.BLOCKED,
            source="unknown",
            target=receipt_id,
            payload="",
            details=details,
            blocked=True
        )
        
        with self._lock:
            self.events.append(event)
            self.blocked_count += 1
        
        return event
    
    def get_recent_events(self, count: int = 50) -> List[SecurityEvent]:
        """Get recent security events"""
        return self.events[-count:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get security statistics"""
        attack_counts = defaultdict(int)
        for event in self.events:
            attack_counts[event.attack_type.value] += 1
        
        return {
            'total_checks': self.total_checks,
            'total_events': len(self.events),
            'blocked_count': self.blocked_count,
            'block_rate': self.blocked_count / max(1, self.total_checks),
            'attack_breakdown': dict(attack_counts),
            'seen_signatures': len(self._seen_signatures)
        }


# Global security monitor instance
_security_monitor: Optional[SecurityMonitor] = None


def get_security_monitor() -> SecurityMonitor:
    """Get or create global security monitor"""
    global _security_monitor
    if _security_monitor is None:
        _security_monitor = SecurityMonitor()
    return _security_monitor


if __name__ == '__main__':
    print("="*60)
    print("SECURITY MODULE TEST")
    print("="*60)
    
    monitor = SecurityMonitor()
    
    # Test SQL injection detection
    print("\n--- SQL Injection Detection ---")
    test_inputs = [
        "SELECT * FROM users",
        "user'; DROP TABLE users--",
        "1 OR 1=1",
        "Normal input text",
        "admin' AND '1'='1",
    ]
    
    for input_str in test_inputs:
        result = monitor.check_input(input_str, source="test", target="api")
        status = "BLOCKED" if result['blocked'] else "ALLOWED"
        print(f"  [{status}] {input_str[:40]}")
        if result['blocked']:
            print(f"           Attacks: {result['attacks']}")
    
    # Test XSS detection
    print("\n--- XSS Detection ---")
    xss_inputs = [
        "<script>alert('xss')</script>",
        "javascript:alert(1)",
        "<img src=x onerror=alert(1)>",
        "Normal text",
    ]
    
    for input_str in xss_inputs:
        result = monitor.check_input(input_str)
        status = "BLOCKED" if result['blocked'] else "ALLOWED"
        print(f"  [{status}] {input_str[:40]}")
    
    # Test rate limiting
    print("\n--- Rate Limiting ---")
    for i in range(15):
        result = monitor.check_rate_limit("test_client")
        if not result['allowed']:
            print(f"  Request {i+1}: BLOCKED - {result['reason']}")
            break
        else:
            print(f"  Request {i+1}: ALLOWED (remaining: {result['remaining']})")
    
    # Test replay detection
    print("\n--- Replay Detection ---")
    sig = "test_signature_12345"
    
    result1 = monitor.check_signature(sig)
    print(f"  First use: {'VALID' if result1['valid'] else 'INVALID'}")
    
    result2 = monitor.check_signature(sig)
    print(f"  Replay attempt: {'VALID' if result2['valid'] else 'BLOCKED - ' + result2.get('reason', '')}")
    
    # Statistics
    print("\n--- Security Stats ---")
    stats = monitor.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
