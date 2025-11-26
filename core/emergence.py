"""
DUAL VERTICAL EMERGENCE CALCULATOR
Patent: US 63/907,140 - FinalBoss Technology

The governance score EMERGES from TWO independent execution paths:
- Vertical #1 executes the task → generates operational score (v1)
- Vertical #2 independently verifies → generates verification score (v2)
- Mathematical formula COMBINES them → produces reliability (R)
- NO central authority decides → governance emerges from the math

Formula: R(v₁, v₂) = v₁ × v₂² + (1-v₁) × ln(1-v₂)
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional
import time


class GovernanceDecision(Enum):
    """Governance decisions that emerge from dual vertical calculation"""
    APPROVE = "APPROVE"      # R > 0.75 - Autonomous approval
    REVIEW = "REVIEW"        # 0.50 < R <= 0.75 - Human review required
    REJECT = "REJECT"        # R <= 0.50 - Blocked


@dataclass
class EmergenceResult:
    """Result of dual vertical emergence calculation"""
    v1: float                           # Vertical 1 operational score
    v2: float                           # Vertical 2 verification score
    term1: float                        # Positive reinforcement term
    term2: float                        # Risk penalty term
    reliability: float                  # Combined emergence score (R)
    decision: GovernanceDecision        # Emerged governance decision
    computation_time_ms: float          # Time to compute
    timestamp: float                    # Unix timestamp


class DualVerticalEmergence:
    """
    Dual Vertical Emergence Calculator
    
    Implements the patented formula:
    R(v₁, v₂) = v₁ × v₂² + (1-v₁) × ln(1-v₂)
    
    term1 = v₁ × v₂²
        Purpose: Positive reinforcement
        When both verticals perform well, R increases
        
    term2 = (1-v₁) × ln(1-v₂)
        Purpose: Risk penalty  
        When uncertainty exists, R decreases
        Note: ln(1-v₂) is negative when v₂ > 0, creating penalty
    """
    
    # Governance thresholds
    APPROVE_THRESHOLD = 0.75
    REVIEW_THRESHOLD = 0.50
    
    # Numerical stability bounds
    MIN_SCORE = 0.001  # Prevent ln(0)
    MAX_SCORE = 0.999  # Prevent ln(0)
    
    def __init__(self):
        self.calculation_count = 0
        self.total_compute_time = 0.0
    
    def calculate(self, v1: float, v2: float) -> EmergenceResult:
        """
        Calculate emergence score from dual vertical inputs.
        
        Args:
            v1: Vertical 1 operational execution score (0.0 to 1.0)
            v2: Vertical 2 verification score (0.0 to 1.0)
            
        Returns:
            EmergenceResult with all computation details
        """
        start_time = time.perf_counter()
        timestamp = time.time()
        
        # Clamp inputs for numerical stability
        v1_clamped = max(self.MIN_SCORE, min(self.MAX_SCORE, v1))
        v2_clamped = max(self.MIN_SCORE, min(self.MAX_SCORE, v2))
        
        # Term 1: Positive reinforcement (reward when both verticals succeed)
        # v₁ × v₂² - higher when both v1 and v2 are high
        term1 = v1_clamped * (v2_clamped ** 2)
        
        # Term 2: Risk penalty (punish uncertainty)
        # (1-v₁) × ln(1-v₂) - always negative (penalty)
        # When v1 is low (high 1-v1) or v2 is low (ln approaches 0), penalty increases
        term2 = (1 - v1_clamped) * math.log(1 - v2_clamped)
        
        # Combined reliability score
        reliability = term1 + term2
        
        # Governance decision emerges from the math
        decision = self._determine_decision(reliability)
        
        # Track computation time
        computation_time_ms = (time.perf_counter() - start_time) * 1000
        self.calculation_count += 1
        self.total_compute_time += computation_time_ms
        
        return EmergenceResult(
            v1=v1,
            v2=v2,
            term1=term1,
            term2=term2,
            reliability=reliability,
            decision=decision,
            computation_time_ms=computation_time_ms,
            timestamp=timestamp
        )
    
    def _determine_decision(self, reliability: float) -> GovernanceDecision:
        """
        Governance decision emerges from reliability score.
        No central authority - pure mathematical consequence.
        """
        if reliability > self.APPROVE_THRESHOLD:
            return GovernanceDecision.APPROVE
        elif reliability > self.REVIEW_THRESHOLD:
            return GovernanceDecision.REVIEW
        else:
            return GovernanceDecision.REJECT
    
    def explain_calculation(self, result: EmergenceResult) -> str:
        """Generate human-readable explanation of the emergence calculation"""
        return f"""
DUAL VERTICAL EMERGENCE CALCULATION
====================================
Input Scores:
  Vertical 1 (Operational): {result.v1:.4f} ({result.v1*100:.1f}%)
  Vertical 2 (Verification): {result.v2:.4f} ({result.v2*100:.1f}%)

Formula: R = v₁×v₂² + (1-v₁)×ln(1-v₂)

Computation:
  term1 = v₁ × v₂²
        = {result.v1:.4f} × {result.v2:.4f}²
        = {result.v1:.4f} × {result.v2**2:.6f}
        = {result.term1:.6f}
        (Positive reinforcement for dual success)

  term2 = (1-v₁) × ln(1-v₂)
        = (1-{result.v1:.4f}) × ln(1-{result.v2:.4f})
        = {1-result.v1:.4f} × ln({1-result.v2:.4f})
        = {1-result.v1:.4f} × {math.log(max(0.001, 1-result.v2)):.6f}
        = {result.term2:.6f}
        (Risk penalty for uncertainty)

  R = term1 + term2
    = {result.term1:.6f} + ({result.term2:.6f})
    = {result.reliability:.6f}
    = {result.reliability*100:.2f}%

EMERGED GOVERNANCE DECISION:
  R = {result.reliability*100:.2f}%
  
  Thresholds:
    > 75%: APPROVE (autonomous)
    > 50%: REVIEW (human check)
    ≤ 50%: REJECT (block)
  
  Decision: {result.decision.value}
  
Computation Time: {result.computation_time_ms:.4f}ms
"""

    def get_stats(self) -> dict:
        """Return computation statistics"""
        avg_time = self.total_compute_time / max(1, self.calculation_count)
        return {
            'total_calculations': self.calculation_count,
            'total_compute_time_ms': self.total_compute_time,
            'average_compute_time_ms': avg_time
        }


class Vertical1:
    """
    Vertical 1: Operational Execution
    
    Executes the actual task and generates an operational success score.
    Independent of Vertical 2.
    """
    
    def __init__(self):
        self.operations = []
        self.total_operations = 0
    
    def execute(self, operation_name: str, operation_func: callable) -> Tuple[any, float]:
        """
        Execute an operation and compute operational score.
        
        Returns:
            Tuple of (operation_result, v1_score)
        """
        start_time = time.perf_counter()
        
        try:
            result = operation_func()
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        execution_time = time.perf_counter() - start_time
        
        # Compute v1 score based on execution success and time
        # Base score from success/failure
        base_score = 0.95 if success else 0.20
        
        # Time penalty (operations taking > 1s get penalized)
        time_penalty = min(0.10, execution_time / 10)
        
        v1_score = max(0.0, min(1.0, base_score - time_penalty))
        
        # Log operation
        operation_record = {
            'name': operation_name,
            'success': success,
            'error': error,
            'execution_time': execution_time,
            'v1_score': v1_score,
            'timestamp': time.time()
        }
        self.operations.append(operation_record)
        self.total_operations += 1
        
        return result, v1_score


class Vertical2:
    """
    Vertical 2: Independent Verification
    
    Independently verifies Vertical 1's execution.
    Generates verification score (v2) without knowledge of v1.
    """
    
    def __init__(self):
        self.verifications = []
        self.total_verifications = 0
    
    def verify(self, operation_name: str, v1_result: any, 
               verification_func: callable = None) -> float:
        """
        Verify an operation result and compute verification score.
        
        Args:
            operation_name: Name of the operation being verified
            v1_result: Result from Vertical 1's execution
            verification_func: Optional custom verification function
            
        Returns:
            v2_score: Verification score (0.0 to 1.0)
        """
        start_time = time.perf_counter()
        
        # Run verification
        if verification_func:
            try:
                is_valid = verification_func(v1_result)
                confidence = 0.95 if is_valid else 0.30
            except Exception:
                is_valid = False
                confidence = 0.10
        else:
            # Default verification: check result exists and is not None
            is_valid = v1_result is not None
            confidence = 0.90 if is_valid else 0.25
        
        verification_time = time.perf_counter() - start_time
        
        # Compute v2 score
        # Base from validation result
        base_score = confidence
        
        # Slight randomization to simulate real-world verification variance
        # (In production, this would be based on actual verification metrics)
        import random
        variance = random.uniform(-0.02, 0.02)
        
        v2_score = max(0.0, min(1.0, base_score + variance))
        
        # Log verification
        verification_record = {
            'operation': operation_name,
            'is_valid': is_valid,
            'confidence': confidence,
            'v2_score': v2_score,
            'verification_time': verification_time,
            'timestamp': time.time()
        }
        self.verifications.append(verification_record)
        self.total_verifications += 1
        
        return v2_score


# Convenience function for quick calculations
def calculate_emergence(v1: float, v2: float) -> EmergenceResult:
    """Quick calculation without instantiating class"""
    calculator = DualVerticalEmergence()
    return calculator.calculate(v1, v2)


if __name__ == '__main__':
    # Demo the emergence calculation
    calculator = DualVerticalEmergence()
    
    # Example from architecture document: v1=0.95, v2=0.97
    result = calculator.calculate(0.95, 0.97)
    print(calculator.explain_calculation(result))
    
    # Test various scenarios
    test_cases = [
        (0.99, 0.99, "Both verticals excellent"),
        (0.80, 0.80, "Both verticals good"),
        (0.60, 0.60, "Both verticals mediocre"),
        (0.40, 0.40, "Both verticals poor"),
        (0.95, 0.50, "V1 good, V2 poor"),
        (0.50, 0.95, "V1 poor, V2 good"),
    ]
    
    print("\n" + "="*60)
    print("EMERGENCE SCORE TEST CASES")
    print("="*60)
    
    for v1, v2, description in test_cases:
        r = calculator.calculate(v1, v2)
        print(f"\n{description}:")
        print(f"  v1={v1:.2f}, v2={v2:.2f} → R={r.reliability:.4f} ({r.reliability*100:.1f}%) → {r.decision.value}")
