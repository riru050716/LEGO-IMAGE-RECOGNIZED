import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

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

# カスタム描画関数
def draw_custom_boxes(image_pil, boxes, names):
    img = image_pil.convert("RGB")
    draw = ImageDraw.Draw(img)
    counts = {}

    # デフォルトフォントの設定（小さめのサイズ）
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = names[cls_id]
        
        # 枠線カラーの設定
        color = CLASS_COLORS.get(class_name, DEFAULT_COLOR)
        
        # 座標の取得 (x1, y1, x2, y2)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        if conf >= 0.60:
            # 【信頼度 60% 以上】文字なし・枠線のみ描画
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            counts[class_name] = counts.get(class_name, 0) + 1
        else:
            # 【信頼度 60% 未満】細い枠線 ＋ 信頼度数値（例: 58%）を表示
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # 信頼度ラベルテキストの作成
            conf_str = f"{int(conf * 100)}%"
            
            # テキストの背景ラベルを描画（見やすさ確保）
            text_pos = (x1, max(0, y1 - 14))
            draw.rectangle([text_pos[0], text_pos[1], text_pos[0] + 32, text_pos[1] + 12], fill=color)
            draw.text((text_pos[0] + 2, text_pos[1]), conf_str, fill=(255, 255, 255), font=font)

    return img, counts

# 画像が準備された場合の処理
if input_image_file is not None:
    image = Image.open(input_image_file)
    st.image(image, caption="対象画像", use_container_width=True)

    if st.button("パーツを識別・カウント", type="primary"):
        with st.spinner("AIでパーツを検出中..."):
            # 推論実行（60%未満の候補を表示・判別するため conf=0.25 で検出）
            results = model(image, conf=0.25, iou=0.5)[0]

            # カスタム描画処理
            res_image, counts = draw_custom_boxes(image, results.boxes, model.names)
            
            # 結果画像表示
            st.image(res_image, caption="検出結果（60%未満は信頼度%を印字）", use_container_width=True)

            # 結果画面表示
            st.markdown("---")
            st.subheader("📊 集計結果")
            st.metric("合計パーツ数 (信頼度 60%以上)", f"{sum(counts.values())} 個")

            # 各クラスの表示（色付きバッジ風）
            if counts:
                st.markdown("### 🏷️ 検出内訳（クラス別カラー）")
                for name, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
                    rgb = CLASS_COLORS.get(name, DEFAULT_COLOR)
                    hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
                    
                    st.markdown(
                        f"<div style='display: flex; align-items: center; margin-bottom: 6px;'>"
                        f"<span style='background-color: {hex_color}; width: 14px; height: 14px; display: inline-block; border-radius: 3px; margin-right: 8px;'></span>"
                        f"<b>{name}</b>: <span style='font-size: 1.1em; font-weight: bold; margin-left: 6px;'>{count} 個</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.warning("パーツが検出されませんでした（またはすべて信頼度60%未満です）。")