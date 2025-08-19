# app.py — Streamlit demo with L/M/R rectangle annotation + 6-class bar chart
import os, tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

import streamlit as st
import matplotlib.pyplot as plt

# ---------------- Config ----------------
# How you want the 6-class bars labeled on the plot (edit to your taste)
# Example labels you showed in your screenshot:
CLASS_NAMES6_DISPLAY = ["1xx","0xx","x1x","x0x","xx1","xx0"]

# If your model's raw output order != the order above, set a reorder index list.
# Leave as None if your model is already in the display order you want.
DISPLAY_REORDER = None          # e.g., [2,3,4,5,0,1]
N_FRAMES = 16
WEIGHTS_FILE = "best_penalty_model.pt"

# If your model logically predicts these 6 classes in the order below,
# we can safely aggregate to L/M/R as [0,1]=Left, [2,3]=Middle, [4,5]=Right.
# If your own order is different, adjust the groups under SIX_TO_THREE_GROUPS.
MODEL_LOGIT_NAMES = ["L-score","L-miss","M-score","M-miss","R-score","R-miss"]
SIX_TO_THREE_GROUPS = {
    "Left":  [0, 1],
    "Middle":[2, 3],
    "Right": [4, 5],
}

TX = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# --------------- Model ------------------
class PenaltyModel(nn.Module):
    def __init__(self, feat_dim=512, lstm_hidden=256, num_classes=6, dropout_prob=0.5, pretrained=True):
        super().__init__()
        try:
            from torchvision.models import ResNet18_Weights
            base = models.resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        except Exception:
            base = models.resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # (B,512,1,1)
        self.lstm = nn.LSTM(input_size=2*feat_dim, hidden_size=lstm_hidden, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, crops, lengths=None):
        # Accept (B,T,2,3,224,224) or (B,T,3,224,224). If latter, duplicate features to make "2".
        if crops.dim() == 5:
            B,T,C,H,W = crops.shape
            x = crops.reshape(B*T, C, H, W)
            f = self.backbone(x).view(B, T, 1, -1)
            f = torch.cat([f, f], dim=2)  # duplicate to simulate 2 objects
        else:
            B,T,O,C,H,W = crops.shape
            x = crops.view(B*T*O, C, H, W)
            f = self.backbone(x).view(B, T, O, -1)
        f = f.reshape(B, T, -1)  # (B,T,2*feat)

        if lengths is None:
            lengths = torch.full((B,), T, dtype=torch.long, device=f.device)

        packed = pack_padded_sequence(f, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        out = self.dropout(h[-1])
        return self.fc(out)

@st.cache_resource
def load_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    model = PenaltyModel(num_classes=6, pretrained=True).to(device)

    w = Path(WEIGHTS_FILE)
    if w.exists():
        sd = torch.load(str(w), map_location=device)
        if isinstance(sd, dict):
            for k in ("state_dict","model","weights"):
                if k in sd and isinstance(sd[k], dict):
                    sd = sd[k]; break
        sd = { (k[7:] if k.startswith("module.") else k): v for k,v in sd.items() }
        model.load_state_dict(sd, strict=False)
    else:
        st.warning(f"Could not find weights file '{WEIGHTS_FILE}'. Running with random weights (UI demo).")
    model.eval()
    return model, device

# ------------- Video helpers -------------
def read_last_n_frames(buf, n=N_FRAMES):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(buf.getbuffer() if hasattr(buf, "getbuffer") else buf.read())
        tmp_path = tmp.name
    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    start = max(0, total - n)
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok: break
        if i >= start: frames.append(f)
        i += 1
    cap.release()
    try: os.remove(tmp_path)
    except: pass
    if 0 < len(frames) < n:
        frames = [frames[0]]*(n-len(frames)) + frames
    return frames[-n:]

def make_tensor_sequence_from_full_frames(frames_bgr):
    xs = []
    for f in frames_bgr:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        xs.append(TX(Image.fromarray(rgb)))
    X = torch.stack(xs, dim=0)   # (T,3,224,224)
    return X.unsqueeze(0)        # (1,T,3,224,224)

# ---------- Prob helpers ----------
def six_to_three_probs(probs6: np.ndarray) -> np.ndarray:
    """Aggregate 6-class probabilities to Left/Middle/Right by summing pairs."""
    l = float(np.sum([probs6[i] for i in SIX_TO_THREE_GROUPS["Left"]]))
    m = float(np.sum([probs6[i] for i in SIX_TO_THREE_GROUPS["Middle"]]))
    r = float(np.sum([probs6[i] for i in SIX_TO_THREE_GROUPS["Right"]]))
    v = np.array([l, m, r], dtype=np.float32)
    s = float(v.sum())
    return v / s if s > 0 else v

# --------- Rectangle overlay (L/M/R) ---------
def annotate_last_frame_rectangles(frame_bgr, probs3, font=cv2.FONT_HERSHEY_SIMPLEX):
    """Draw Left/Middle/Right boxes with percentages (probs3 length=3)."""
    frame = frame_bgr.copy()
    h, w = frame.shape[:2]
    y1, y2 = int(0.22*h), int(0.68*h)  # tweak as needed
    thirds = [0, w//3, (2*w)//3, w]
    labels = ["Left", "Middle", "Right"]
    colors = [(60,140,255), (40,220,255), (180,220,255)]  # BGR

    for i, lbl in enumerate(labels):
        x1, x2 = thirds[i], thirds[i+1]
        color = colors[i]
        cv2.rectangle(frame, (x1+3, y1), (x2-3, y2), color, 3)
        # Title
        (tw, th), _ = cv2.getTextSize(lbl, font, 0.8, 2)
        tx = x1 + (x2-x1)//2 - tw//2
        ty = max(24, y1-10)
        cv2.putText(frame, lbl, (tx, ty), font, 0.8, (255,255,255), 2, cv2.LINE_AA)
        # Probability
        pct = f"{int(round(float(probs3[i]) * 100))}%"
        (pw, ph), _ = cv2.getTextSize(pct, font, 1.0, 2)
        px = x1 + (x2-x1)//2 - pw//2
        py = y1 + (y2-y1)//2 + ph//2
        cv2.putText(frame, pct, (px, py), font, 1.0, (255,255,255), 2, cv2.LINE_AA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# ---------------- UI ----------------
st.set_page_config(page_title="Penalty Kick Predictor", layout="centered")
st.title("Penalty Kick Predictor")
st.caption("Upload a short pre‑kick video; we’ll show the last frame with L/M/R overlay and bar charts.")

uploaded = st.file_uploader("MP4 video (≤ ~10s)", type=["mp4","mov","m4v"])
if uploaded:
    st.video(uploaded)

    # Grab frames and build tensor
    frames = read_last_n_frames(uploaded, N_FRAMES)
    if not frames:
        st.error("Could not read frames from the video.")
        st.stop()

    X = make_tensor_sequence_from_full_frames(frames)
    model, device = load_model()
    X = X.to(device)
    lengths = torch.tensor([X.shape[1]], device=device)

    # Inference
    with torch.no_grad():
        try:
            logits = model(X, lengths)
        except TypeError:
            logits = model(X)
        probs6 = torch.softmax(logits, dim=1).cpu().numpy()[0]   # (6,)

    # ----- 6-class bar chart (display order you prefer) -----
    if DISPLAY_REORDER is None:
        probs6_disp = probs6
        names6_disp = CLASS_NAMES6_DISPLAY
    else:
        probs6_disp = probs6[DISPLAY_REORDER]
        names6_disp = [CLASS_NAMES6_DISPLAY[i] for i in range(len(CLASS_NAMES6_DISPLAY))]
    fig6 = plt.figure(figsize=(6,3))
    plt.bar(names6_disp, probs6_disp)
    plt.ylim(0, 1)
    plt.ylabel("Probability")
    plt.title("Penalty Kick Prediction (6 classes)")
    plt.tight_layout()
    st.pyplot(fig6, use_container_width=True)

    # ----- Annotated last frame (L/M/R) -----
    probs3 = six_to_three_probs(probs6)
    ann_rgb = annotate_last_frame_rectangles(frames[-1], probs3)
    st.image(ann_rgb, caption="Directional probabilities (L/M/R)", use_container_width=True)

    # ----- 3-class bar chart (L/M/R) -----
    fig3 = plt.figure(figsize=(6,3))
    plt.bar(["Left","Middle","Right"], probs3)
    plt.ylim(0,1)
    plt.ylabel("Probability")
    plt.title("Directional Probabilities (L/M/R)")
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)

    # Optional raw dumps
    st.write("6-class (display order):", {n: float(p) for n,p in zip(CLASS_NAMES6_DISPLAY, probs6_disp)})
    st.write("3-class (L/M/R):", {"Left": float(probs3[0]), "Middle": float(probs3[1]), "Right": float(probs3[2])})
else:
    st.info("Upload a short penalty video to get predictions.")
