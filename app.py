import streamlit as st
from ultralytics import YOLO
from PIL import Image

# 画面設定
st.set_page_config(page_title="LEGO SPIKE パーツカウンター", page_icon="🧩", layout="centered")

st.title("🧩 LEGOパーツ 自動カウント")
st.write("写真を撮影またはアップロードすると、AIがパーツの種類と個数を判定します。")

# モデルの読み込み (キャッシュ化して高速化)
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error("モデルファイル 'best.pt' が見つかりません。プロジェクト直下に配置してください。")
    st.stop()

# 画像アップロード（スマホのカメラ撮影対応）
uploaded_file = st.file_uploader("パーツの写真をえらぶ", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # アップロード画像表示
    image = Image.open(uploaded_file)
    st.image(image, caption="アップロード画像", use_container_width=True)

    if st.button("パーツを識別・カウント", type="primary"):
        with st.spinner("AIでパーツを検出中..."):
            # 推論実行
            results = model(image)[0]

            # 検出結果画像の描画・表示
            res_plotted = results.plot()
            st.image(res_plotted, caption="検出結果", use_container_width=True)

            # 個数集計
            counts = {}
            for box in results.boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                counts[class_name] = counts.get(class_name, 0) + 1

            # 結果画面表示
            st.markdown("---")
            st.subheader("📊 集計結果")
            st.metric("合計パーツ数", f"{sum(counts.values())} 個")

            if counts:
                # 検出パーツの一覧表示
                for name, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
                    st.write(f"- **{name}**: {count} 個")
            else:
                st.warning("パーツが検出されませんでした。")