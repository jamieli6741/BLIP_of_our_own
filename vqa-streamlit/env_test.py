# test_env.py
import sys
print(f"Python版本: {sys.version}")

try:
    import torch
    print(f"✓ PyTorch版本: {torch.__version__}")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA版本: {torch.version.cuda}")
except ImportError:
    print("✗ PyTorch未安装")

try:
    import transformers
    print(f"✓ Transformers版本: {transformers.__version__}")
except ImportError:
    print("✗ Transformers未安装")

try:
    import streamlit as st
    print(f"✓ Streamlit版本: {st.__version__}")
except ImportError:
    print("✗ Streamlit未安装")

try:
    from PIL import Image
    import PIL
    print(f"✓ Pillow版本: {PIL.__version__}")
except ImportError:
    print("✗ Pillow未安装")

# 测试BLIP模型是否可以加载
print("\n测试BLIP模型加载...")
try:
    from transformers import BlipProcessor, BlipForQuestionAnswering
    print("✓ BLIP模型类可以导入")
except ImportError as e:
    print(f"✗ BLIP模型类导入失败: {e}")
    print("  尝试升级transformers: pip install transformers>=4.35.0")
