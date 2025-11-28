#!/usr/bin/env python3
"""Railway start script for JUGGERNAUT"""
import os
import sys

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.server import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = '0.0.0.0'

    print("="*60)
    print("  JUGGERNAUT PLATFORM - RAILWAY DEPLOYMENT")
    print("  FinalBoss Technology - Abraham Manzano")
    print("="*60)
    print(f"Starting server on {host}:{port}")
    print("="*60)

    app.run(host=host, port=port, debug=False)
