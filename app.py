import json
from pathlib import Path

import torch
import streamlit as st
from PIL import Image

from src.data import get_transforms
from src.models import build_model


@st.cache_resource
def load_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = build_model(ckpt["model_name"], len(ckpt["classes"]), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt, device


st.set_page_config(page_title="Garbage Classifier", layout="centered")
st.title("Garbage Classification")

checkpoint_path = st.sidebar.text_input("Checkpoint", "outputs/checkpoints/best.pt")
uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp"])

if uploaded and Path(checkpoint_path).exists():
    model, ckpt, device = load_model(checkpoint_path)
    classes = ckpt["classes"]
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Input image", use_container_width=True)

    transform = get_transforms(ckpt.get("image_size", 224), train=False)
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = model(tensor).softmax(dim=1)[0].cpu()
    top3 = torch.topk(probs, k=min(3, len(classes)))

    st.subheader("Top predictions")
    for idx, prob in zip(top3.indices.tolist(), top3.values.tolist()):
        st.write(f"{classes[idx]}: {prob:.2%}")
elif uploaded:
    st.error(f"Checkpoint not found: {checkpoint_path}")
else:
    st.info("Upload an image to classify waste type.")
