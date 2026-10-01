import torch
import sys

print("=" * 50)
print(f"Python バージョン: {sys.version}")
print(f"PyTorch バージョン: {torch.__version__}")
print("=" * 50)

# PyTorchのCUDA対応チェック
cuda_available = torch.cuda.is_available()
print(f"CUDA 使用可能フラグ: {cuda_available}")

if cuda_available:
    print(f"ビルド時CUDAバージョン: {torch.version.cuda}")
    print(f"認識されたGPU数: {torch.cuda.device_count()}")
    print(f"使用GPU名: {torch.cuda.get_device_name(0)}")
    print("STATUS: GPU学習が利用可能です！")
else:
    print("STATUS: PyTorchがGPUを認識していません（CPUモードになります）。")
    if "+cpu" in torch.__version__:
        print("原因: CPU専用版のPyTorchがインストールされています。")
    elif torch.version.cuda is None:
        print("原因: インストールされているPyTorchにCUDAサポートが含まれていません。")
print("=" * 50)