"""运行前检查：GPU 是否可用、路径是否正确。"""
import os
import sys

import config

print("=" * 50)
print("1. 路径检查")
print("=" * 50)
paths = {
    "模型目录": config.MODEL_PATH,
    "训练数据": config.TRAIN_PATH,
    "验证数据": config.VAL_PATH,
}
all_ok = True
for name, path in paths.items():
    exists = os.path.exists(path)
    status = "OK" if exists else "缺失"
    print(f"  [{status}] {name}: {path}")
    if not exists:
        all_ok = False

print()
print("=" * 50)
print("2. PyTorch / GPU 检查")
print("=" * 50)
try:
    import torch

    print(f"  PyTorch 版本: {torch.__version__}")
    print(f"  CUDA 编译支持: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        x = torch.randn(2, 2, device="cuda")
        y = x @ x
        print(f"  GPU 矩阵运算测试: 通过 (device={y.device})")
    else:
        print("  警告: 当前 PyTorch 无法使用 GPU")
        print("  请安装带 CUDA 的 PyTorch，例如:")
        print("    pip install torch --index-url https://download.pytorch.org/whl/cu124")
        all_ok = False
except ImportError:
    print("  错误: 未安装 torch，请先 pip install torch")
    all_ok = False

print()
print("=" * 50)
print("3. Transformers 检查")
print("=" * 50)
try:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH, local_files_only=True)
    print(f"  Tokenizer 加载: OK (vocab_size={tokenizer.vocab_size})")
except Exception as e:
    print(f"  Tokenizer 加载失败: {e}")
    all_ok = False

print()
if all_ok:
    print("全部检查通过，可以运行: python train.py")
else:
    print("存在问题，请先修复后再训练。")
    sys.exit(1)
