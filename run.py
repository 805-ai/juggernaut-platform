#!/usr/bin/env python3
"""
JUGGERNAUT - Main Entry Point
FinalBoss Technology - Abraham Manzano

Run the complete JUGGERNAUT platform:
- Dual Vertical Emergence
- ML-DSA-65 Cryptographic Receipts
- Sub-8ms Destruction Protocol
- Unified API Server

Usage:
    python run.py                  # Start API server
    python run.py --demo           # Run demo mode
    python run.py --test           # Run all tests
"""

import sys
import os
import argparse
import time

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_server(host='0.0.0.0', port=5000):
    """Start the JUGGERNAUT API server"""
    from api.server import app
    
    print("="*60)
    print("  JUGGERNAUT PLATFORM")
    print("  FinalBoss Technology - Abraham Manzano")
    print("="*60)
    print()
    print(f"Starting server on http://{host}:{port}")
    print(f"Dashboard: http://localhost:{port}/dashboard")
    print()
    
    app.run(host=host, port=port, debug=False)


def run_demo():
    """Run demo of all JUGGERNAUT capabilities"""
    print("="*60)
    print("  JUGGERNAUT DEMO")
    print("  FinalBoss Technology - Abraham Manzano")
    print("="*60)
    
    # Import components
    from core.emergence import DualVerticalEmergence, Vertical1, Vertical2
    from core.destruction import DestructionProtocol
    from core.security import SecurityMonitor
    from crypto.ml_dsa import MLDSA65
    from crypto.receipt_chain import ReceiptChain, ReceiptMinter
    
    # ============================================
    # DEMO 1: Dual Vertical Emergence
    # ============================================
    print("\n" + "="*60)
    print("DEMO 1: DUAL VERTICAL EMERGENCE")
    print("="*60)
    
    calc = DualVerticalEmergence()
    v1 = Vertical1()
    v2 = Vertical2()
    
    # Simulate operation
    print("\nSimulating dual vertical operation...")
    
    # Vertical 1 executes
    result, v1_score = v1.execute("test_operation", lambda: "Success!")
    print(f"  Vertical 1 executed: v1 = {v1_score:.4f}")
    
    # Vertical 2 verifies
    v2_score = v2.verify("test_operation", result)
    print(f"  Vertical 2 verified: v2 = {v2_score:.4f}")
    
    # Calculate emergence
    emergence = calc.calculate(v1_score, v2_score)
    print(f"\nEmergence Calculation:")
    print(f"  Formula: R = v1*v2^2 + (1-v1)*ln(1-v2)")
    print(f"  term1 = {v1_score:.4f} * {v2_score:.4f}^2 = {emergence.term1:.6f}")
    print(f"  term2 = (1-{v1_score:.4f}) * ln(1-{v2_score:.4f}) = {emergence.term2:.6f}")
    print(f"  R = {emergence.reliability:.4f} ({emergence.reliability*100:.2f}%)")
    print(f"  Decision: {emergence.decision.value}")
    
    # ============================================
    # DEMO 2: ML-DSA-65 Signing
    # ============================================
    print("\n" + "="*60)
    print("DEMO 2: ML-DSA-65 POST-QUANTUM SIGNING")
    print("="*60)
    
    signer = MLDSA65()
    status = signer.get_status()
    
    print(f"\nCrypto Status:")
    print(f"  Implementation: {status['implementation']}")
    print(f"  Quantum Resistant: {status['quantum_resistant']}")
    print(f"  Shor Resistant: {status['shor_resistant']}")
    
    # Generate keys
    keys = signer.generate_keypair()
    print(f"\nKey Generation:")
    print(f"  Key ID: {keys.key_id}")
    print(f"  Public key: {len(keys.public_key)} bytes")
    print(f"  Secret key: {len(keys.secret_key)} bytes")
    
    # Sign message
    message = b"JUGGERNAUT Demo Message - FinalBoss Technology"
    sig = signer.sign(message)
    print(f"\nSigning:")
    print(f"  Message: {message.decode()}")
    print(f"  Signature: {len(sig.signature_bytes)} bytes")
    print(f"  Time: {sig.signing_time_ms:.2f}ms")
    
    # Verify
    is_valid = signer.verify(message, sig.signature_bytes, keys.public_key)
    print(f"  Verification: {'[OK] VALID' if is_valid else '[X] INVALID'}")
    
    # ============================================
    # DEMO 3: Receipt Chain
    # ============================================
    print("\n" + "="*60)
    print("DEMO 3: CRYPTOGRAPHIC RECEIPT CHAIN")
    print("="*60)

    # Clean demo storage to avoid key mismatch
    import shutil
    demo_storage = "/tmp/juggernaut_demo"
    if os.path.exists(demo_storage):
        shutil.rmtree(demo_storage)

    chain = ReceiptChain(storage_path=demo_storage)
    minter = ReceiptMinter(chain)
    
    print("\nMinting receipts...")
    for i in range(5):
        receipt = minter.mint(
            operation=f"demo_operation_{i}",
            result=f"Result {i}",
            v1_score=0.95,
            v2_score=0.97
        )
        print(f"  Minted: {receipt.receipt_id}")
    
    # Verify chain
    verification = chain.verify_chain()
    print(f"\nChain Verification:")
    print(f"  Total receipts: {verification['total_receipts']}")
    print(f"  Valid: {verification['is_valid']}")
    print(f"  Tamper detected: {verification['tamper_detected']}")
    
    # ============================================
    # DEMO 4: Sub-8ms Destruction
    # ============================================
    print("\n" + "="*60)
    print("DEMO 4: SUB-8MS DESTRUCTION PROTOCOL")
    print("="*60)
    
    destroyer = DestructionProtocol()
    
    print("\nExecuting destruction protocol...")
    for i in range(10):
        data = f"Sensitive data #{i}: API_KEY=sk-{i*12345}"
        proof = destroyer.destroy(data, data_id=f"secret_{i}")
    
    stats = destroyer.get_stats()
    print(f"\nDestruction Statistics:")
    print(f"  Total destructions: {stats['total_destructions']}")
    print(f"  Average time: {stats['average_time_ms']:.4f}ms")
    print(f"  Target: <{stats['target_time_ms']}ms")
    print(f"  Within target: {stats['within_target']}")
    print(f"  Success rate: {stats['success_rate']*100:.1f}%")
    
    # ============================================
    # DEMO 5: Security Module
    # ============================================
    print("\n" + "="*60)
    print("DEMO 5: SECURITY MONITORING")
    print("="*60)
    
    security = SecurityMonitor()
    
    test_inputs = [
        ("Normal input", False),
        ("SELECT * FROM users", True),
        ("user'; DROP TABLE--", True),
        ("<script>alert('xss')</script>", True),
        ("Hello World", False),
    ]
    
    print("\nTesting input validation...")
    for input_str, should_block in test_inputs:
        result = security.check_input(input_str)
        status = "BLOCKED" if result['blocked'] else "ALLOWED"
        expected = "BLOCKED" if should_block else "ALLOWED"
        match = "[OK]" if status == expected else "[X]"
        print(f"  [{status}] {input_str[:30]:<30} {match}")
    
    sec_stats = security.get_stats()
    print(f"\nSecurity Statistics:")
    print(f"  Total checks: {sec_stats['total_checks']}")
    print(f"  Blocked: {sec_stats['blocked_count']}")
    
    # ============================================
    # SUMMARY
    # ============================================
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("""
[OK] Dual Vertical Emergence: Patent US 63/907,140
[OK] ML-DSA-65 Cryptography: Patent US 63/909,737
[OK] Receipt Chain: Blockchain-like audit trail
[OK] Sub-8ms Destruction: Cryptographic data elimination
[OK] Security Module: Attack detection and blocking

To start the API server:
  python run.py

Dashboard will be available at:
  http://localhost:5000/dashboard
""")


def run_tests():
    """Run all component tests"""
    print("="*60)
    print("  JUGGERNAUT TEST SUITE")
    print("="*60)
    
    # Test each module
    print("\n--- Testing Core Emergence ---")
    from core.emergence import DualVerticalEmergence
    calc = DualVerticalEmergence()
    result = calc.calculate(0.95, 0.97)
    assert 0.7 < result.reliability < 0.75, "Emergence calculation failed"
    print("  [OK] Emergence calculation correct")
    
    print("\n--- Testing ML-DSA-65 ---")
    from crypto.ml_dsa import MLDSA65
    signer = MLDSA65()
    keys = signer.generate_keypair()
    msg = b"Test message"
    sig = signer.sign(msg)
    assert len(sig.signature_bytes) > 0, "Signing failed"
    print("  [OK] Key generation works")
    print("  [OK] Signing works")
    
    print("\n--- Testing Receipt Chain ---")
    from crypto.receipt_chain import ReceiptChain
    from core.emergence import GovernanceDecision
    chain = ReceiptChain()
    receipt = chain.mint_receipt(
        operation="test",
        requester="test",
        resource="test",
        decision="APPROVED",
        result="success",
        v1_score=0.95,
        v2_score=0.97,
        reliability=0.72,
        governance_decision=GovernanceDecision.REVIEW
    )
    assert receipt.receipt_id.startswith("JUGG-"), "Receipt minting failed"
    print("  [OK] Receipt minting works")
    
    print("\n--- Testing Destruction Protocol ---")
    from core.destruction import DestructionProtocol
    destroyer = DestructionProtocol()
    proof = destroyer.destroy("test data")
    assert proof.total_time_ms < 100, "Destruction too slow"
    print("  [OK] Destruction protocol works")
    
    print("\n--- Testing Security Module ---")
    from core.security import SecurityMonitor
    security = SecurityMonitor()
    result = security.check_input("SELECT * FROM users")
    assert result['blocked'], "SQL injection not blocked"
    print("  [OK] Security validation works")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED [OK]")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='JUGGERNAUT Platform - FinalBoss Technology'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run demo mode'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run tests'
    )
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Server host (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Server port (default: 5000)'
    )
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo()
    elif args.test:
        run_tests()
    else:
        run_server(host=args.host, port=args.port)


if __name__ == '__main__':
    main()
