import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageEnhance
import numpy as np

# 画面設定
st.set_page_config(page_title="LEGO SPIKE パーツカウンター", page_icon="🧩", layout="centered")

st.title("🧩 LEGOパーツ 自動カウント")
st.write("写真を撮影またはアップロードすると、AIがパーツの種類と個数を判定します。")

# モデルの読み込み (キャッシュ化)
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error("モデルファイル 'best.pt' が見つかりません。プロジェクト直下に配置してください。")
    st.stop()

# --- クラス別の枠線カラー定義（RGB） ---
CLASS_COLORS = {
    "青ピン": (0, 102, 255),       # 青
    "橙ピン": (255, 128, 0),       # 橙
    "灰色ピン(短)": (128, 128, 128), # 灰（明るめ）
    "灰色ピン(長)": (80, 80, 80),    # 灰（暗め）
    "黒ピン(丸)": (180, 0, 255),    # 紫（黒識別用）
    "黒ピン(短)": (255, 0, 102),    # ビビッドピンク
    "黒ピン(長)": (0, 204, 153),    # エメラルドグリーン
}
DEFAULT_COLOR = (255, 255, 0) # 定義外の代替色（黄）

# --- 入力方法の選択 ---
input_type = st.radio(
    "画像の準備方法を選んでください",
    ["📷 直接カメラで撮影", "📁 ファイルから選択"],
    horizontal=True
)

input_image_file = None

if input_type == "📷 直接カメラで撮影":
    input_image_file = st.camera_input("パーツを画面枠に収めて撮影してください")
else:
    input_image_file = st.file_uploader("パーツの写真をえらぶ", type=["jpg", "jpeg", "png"])

# 画像処理・描画関数
def draw_custom_boxes(image_pil, boxes, names):
    # 元画像の複製（RGBA変換）
    img = image_pil.convert("RGBA")
    
    # 未確定パーツ（75%未満）を明るく強調するためのベース画像
    bright_base = ImageEnhance.Brightness(img.convert("RGB")).enhance(1.3).convert("RGBA")
    
    # 描画レイヤー作成
    draw = ImageDraw.Draw(img)
    
    high_conf_counts = {}
    low_conf_counts = {}

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = names[cls_id]
        
        # 枠線カラーの設定
        color = CLASS_COLORS.get(class_name, DEFAULT_COLOR)
        
        # 座標の取得 (x1, y1, x2, y2)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        if conf >= 0.75:
            # 【信頼度 75% 以上】太い実線枠で明確に囲む
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            high_conf_counts[class_name] = high_conf_counts.get(class_name, 0) + 1
        else:
            # 【信頼度 75% 未満】明るく表示 ＋ 細い枠線
            crop_bright = bright_base.crop((x1, y1, x2, y2))
            img.paste(crop_bright, (x1, y1))
            
            # 明るくした領域に細い枠を描画
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            low_conf_counts[class_name] = low_conf_counts.get(class_name, 0) + 1

    return img.convert("RGB"), high_conf_counts, low_conf_counts

# 画像が準備された場合の処理
if input_image_file is not None:
    image = Image.open(input_image_file)
    st.image(image, caption="対象画像", use_container_width=True)

    if st.button("パーツを識別・カウント", type="primary"):
        with st.spinner("AIでパーツを検出中..."):
            # 推論実行（低信頼度の検出も保持して後から色分けるため conf=0.25）
            results = model(image, conf=0.25, iou=0.5)[0]

            # カスタム枠描画処理
            res_image, high_counts, low_counts = draw_custom_boxes(image, results.boxes, model.names)
            
            # 結果画像描画・表示
            st.image(res_image, caption="検出結果（枠線カラー識別）", use_container_width=True)

            # 結果画面表示
            st.markdown("---")
            st.subheader("📊 集計結果")
            
            total_high = sum(high_counts.values())
            total_low = sum(low_counts.values())
            
            st.metric("確定パーツ数 (信頼度 75%以上)", f"{total_high} 個")
            if total_low > 0:
                st.caption(f"※ 信頼度75%未満の候補パーツ: {total_low} 個（画像上で明るく表示）")

            # 各クラスの表示（色付きバッジ風）
            if high_counts or low_counts:
                st.markdown("### 🏷️ 検出内訳（クラス別カラー）")
                
                # 検出されたすべてのクラス名を取得
                all_classes = sorted(list(set(high_counts.keys()) | set(low_counts.keys())))
                
                for cls in all_classes:
                    high_num = high_counts.get(cls, 0)
                    low_num = low_counts.get(cls, 0)
                    
                    rgb = CLASS_COLORS.get(cls, DEFAULT_COLOR)
                    hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
                    
                    # カラーマークと個数の書き出し
                    st.markdown(
                        f"<div style='display: flex; align-items: center; margin-bottom: 6px;'>"
                        f"<span style='background-color: {hex_color}; width: 14px; height: 14px; display: inline-block; border-radius: 3px; margin-right: 8px;'></span>"
                        f"<b>{cls}</b>: <span style='font-size: 1.1em; font-weight: bold; margin-left: 6px;'>{high_num} 個</span>"
                        f"{f' <span style=\"color: #888;\">（※候補: {low_num}個）</span>' if low_num > 0 else ''}"
                        f"</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.warning("パーツが検出されませんでした。")