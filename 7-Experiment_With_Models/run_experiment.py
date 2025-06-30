#!/usr/bin/env python3
"""
Main entry point for the refactored ML experiment framework
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiment.experiment_runner import main

if __name__ == "__main__":
    main() 