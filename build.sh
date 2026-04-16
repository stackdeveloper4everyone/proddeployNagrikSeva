#!/bin/bash
set -e

# Install dependencies with binary preference to avoid compilation issues
pip install --prefer-binary -r requirements.txt