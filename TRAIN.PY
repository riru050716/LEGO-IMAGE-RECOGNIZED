import torch
from ultralytics import YOLO

def main():
    # 1. RTX 3060 (GPU) の認識確認
    print(f"CUDA 使用可能: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("警告: GPUが認識されていません。CPUで実行されます。")
        return

    # 2. モデルの選択
    # 【パターンA】Roboflowから「YOLOv8」形式でダウンロードした場合（物体検出）
    model_name = "yolov8s.pt"
    
    # 【パターンB】Roboflowから「YOLOv8-seg」形式でダウンロードした場合（セグメンテーション）
    # model_name = "yolov8s-seg.pt"

    print(f"使用モデル: {model_name}")
    model = YOLO(model_name)

    # 3. RTX 3060 での学習実行
    results = model.train(
        data="C:\\Users\\daich\\Downloads\\lego_dataset\\data.yaml",    # dat`a.yaml への相対パス
        epochs=80,                   # 学習回数（20種類のパーツがあるので60~100程度推奨）
        imgsz=640,                   # 画像サイズ
        batch=16,                    # RTX 3060 (VRAM 12GB/6GB) に適したバッチサイズ
        device=0,                    # GPU:0 (RTX 3060) を指定
        workers=4,                   # データロード用のCPUスレッド数
        name="lego_brick_3060",      # 出力フォルダ名
        exist_ok=True                # 上書きを許可
    )

    print("学習が完了しました！")
    print("生成されたモデル: runs/detect/lego_brick_3060/weights/best.pt")

if __name__ == "__main__":
    main()