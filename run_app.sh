#!/bin/bash
cd "$(dirname "$0")"
echo "🚀 Starting ASL Translation System (Camera Test)..."
echo "Using Python environment: .venv39"
./.venv39/bin/streamlit run app_simple.py
echo "✅ App stopped."
