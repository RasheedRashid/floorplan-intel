"""Interactive viewer: upload a drawing, see detected rooms, confidence and review flags.

    streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fpintel.inference import Predictor  # noqa: E402
from fpintel.render import analyse, draw_overlay, load_drawing  # noqa: E402

st.set_page_config(page_title="Floor-plan intelligence prototype", layout="wide")
st.title("Floor-plan → structured building data")
st.caption("Research prototype. Not validated for underwriting, valuation or any real decision.")

with st.sidebar:
    model_path = st.text_input("Checkpoint", "checkpoints/best.pth")
    mc_samples = st.slider("MC-dropout samples", 1, 32, 8, help="More samples = better uncertainty, slower")
    threshold = st.slider("Review threshold", 0.0, 1.0, 0.8, 0.05,
                          help="Rooms with calibrated confidence below this are flagged for a human")


@st.cache_resource
def get_predictor(path: str) -> Predictor:
    return Predictor(path, Path(path).parent / "calibration.json")


upload = st.file_uploader("Floor-plan image or PDF", type=["png", "jpg", "jpeg", "pdf"])
if upload is None:
    st.info("Upload a floor plan to begin.")
    st.stop()

if not Path(model_path).exists():
    st.error(f"Checkpoint not found: {model_path}. Train a model first (see README).")
    st.stop()

image = load_drawing(upload.getvalue(), upload.name)
with st.spinner("Analysing drawing..."):
    result, labels = analyse(get_predictor(model_path), image, mc_samples, threshold)

s = result["summary"]
c1, c2, c3 = st.columns(3)
c1.metric("Rooms detected", s["room_count"])
c2.metric("Flagged for review", s["rooms_flagged_for_review"])
c3.metric("Temperature (T)", f"{result['model']['temperature']:.2f}")

left, right = st.columns(2)
left.image(image, caption="Input drawing", width="stretch")
right.image(draw_overlay(image, labels, result), caption="Detected rooms (red = needs review)",
            width="stretch")

st.subheader("Rooms")
st.dataframe([{k: r.get(k) for k in ["id", "type", "confidence", "needs_review", "area_fraction", "adjacent_to"]}
              for r in result["rooms"]], width="stretch")
st.download_button("Download JSON", json.dumps(result, indent=2), file_name="building_data.json",
                   mime="application/json")
with st.expander("Raw JSON"):
    st.json(result)
