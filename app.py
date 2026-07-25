import os
import re
import html
import json
import math
import joblib
import emoji
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModel

app = Flask(__name__)

# ==========================================
# 1. SETUP CONFIGURATION & GLOBALS
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("model_configuration.json", "r", encoding="utf-8") as f:
    config = json.load(f)

LABELS = config["labels"]
MAX_LENGTH = config["max_length"]
NRC_FEATURE_COLUMNS = config["nrc_feature_columns"]
NRC_CATEGORY = config["nrc_categories"]
USE_SLANG_NORMALIZATION = config["use_slang_normalization"]
USE_EMOJI_DEMOJIZE = config["use_emoji_demojize"]

# Load NRC artifacts
nrc_scaler = joblib.load("nrc_standard_scaler.joblib")
nrc_data = joblib.load("nrc_lookup_and_metadata.joblib")
nrc_lookup = nrc_data["nrc_lookup"]

# ==========================================
# 2. PREPROCESSING FUNCTIONS & REGEX
# ==========================================
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
MENTION_PATTERN = re.compile(r"(?<!\w)@\w+", flags=re.UNICODE)
WORD_PATTERN = re.compile(r"\b[\w]+\b", flags=re.UNICODE)

# Load Slang Dictionary
slang_map = {}
if USE_SLANG_NORMALIZATION and os.path.exists("kamuskatabaku.xlsx"):
    kamus_slang = pd.read_excel("kamuskatabaku.xlsx", dtype=str)
    kamus_slang.columns = kamus_slang.columns.astype(str).str.lower().str.strip()
    kamus_slang = kamus_slang.rename(columns={"tidak_baku": "slang", "kata_baku": "formal"})
    kamus_slang = kamus_slang[["slang", "formal"]].dropna()
    slang_map = dict(zip(kamus_slang["slang"].str.lower().str.strip(), kamus_slang["formal"].str.lower().str.strip()))

def ganti_slang_dalam_teks(text):
    if not USE_SLANG_NORMALIZATION:
        return text
    def replacement(match):
        token_asli = match.group(0)
        if token_asli in {"HTTPURL", "USER", "URLTOKEN", "USERTOKEN"}:
            return token_asli
        return slang_map.get(token_asli.lower(), token_asli)
    return WORD_PATTERN.sub(replacement, text)

def preprocessing_indobert(text):
    text = html.unescape(str(text)).lower()
    text = URL_PATTERN.sub(" URLTOKEN ", text)
    text = MENTION_PATTERN.sub(" USERTOKEN ", text)
    text = ganti_slang_dalam_teks(text)
    text = text.replace("URLTOKEN", "HTTPURL").replace("USERTOKEN", "@USER")
    if USE_EMOJI_DEMOJIZE:
        text = emoji.demojize(text, language="en", delimiters=(" ", " "))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalisasi_teks_nrc(text):
    text = html.unescape(str(text)).lower()
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    text = text.replace("#", " ")
    tokens = re.findall(r"[a-zA-ZÀ-ÖØ-öø-ÿ]+", text)
    normalized_tokens = []
    for token in tokens:
        replacement = slang_map.get(token, token) if USE_SLANG_NORMALIZATION else token
        normalized_tokens.extend(re.findall(r"[a-zA-ZÀ-ÖØ-öø-ÿ]+", str(replacement).lower()))
    return " ".join(normalized_tokens)

def ekstraksi_fitur_nrc(text):
    text_nrc = normalisasi_teks_nrc(text)
    tokens = text_nrc.split()
    emotion_counts = np.zeros(len(NRC_CATEGORY), dtype=np.float32)
    detected_words = 0
    for token in tokens:
        emotion_vector = nrc_lookup.get(token)
        if emotion_vector is not None:
            emotion_counts += emotion_vector
            detected_words += 1
    total_words = len(tokens)
    coverage = (detected_words / total_words if total_words > 0 else 0.0)
    
    features = {f"nrc_{cat}": float(emotion_counts[idx]) for idx, cat in enumerate(NRC_CATEGORY)}
    features.update({
        "jumlah_kata": float(total_words),
        "kata_terdeteksi_nrc": float(detected_words),
        "cakupan_nrc": float(coverage)
    })
    return features

# ==========================================
# 3. PYTORCH MODEL CLASSES
# ==========================================
class FullWidthCNN2DOptimized(nn.Module):
    def __init__(self, out_channels, kernel_height, feature_width, bias=True):
        super().__init__()
        self.feature_width = feature_width
        self.weight = nn.Parameter(torch.empty(out_channels, 1, kernel_height, feature_width))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.size(1) * self.weight.size(2) * self.weight.size(3)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, sequence_features):
        input_1d = sequence_features.transpose(1, 2)
        weight_1d = self.weight[:, 0, :, :].transpose(1, 2).contiguous()
        return F.conv1d(input=input_1d, weight=weight_1d, bias=self.bias, stride=1, padding=0)

class IndoBERTBaseP1NRCFusionCNN2D(nn.Module):
    def __init__(self, bert_hidden_size, num_classes, nrc_input_dim, nrc_projection_dim=32, filter_sizes=(2, 3, 4, 5), num_filters=128, dropout=0.5, nrc_projection_dropout=0.1):
        super().__init__()
        self.filter_sizes = tuple(filter_sizes)
        self.nrc_projection = nn.Sequential(
            nn.Linear(nrc_input_dim, nrc_projection_dim),
            nn.GELU(),
            nn.LayerNorm(nrc_projection_dim),
            nn.Dropout(nrc_projection_dropout)
        )
        fusion_width = bert_hidden_size + nrc_projection_dim
        self.convolutions = nn.ModuleList([
            FullWidthCNN2DOptimized(out_channels=num_filters, kernel_height=fs, feature_width=fusion_width) for fs in filter_sizes
        ])
        jumlah_fitur_cnn = num_filters * len(filter_sizes)
        self.layer_norm = nn.LayerNorm(jumlah_fitur_cnn)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(jumlah_fitur_cnn, num_classes)

    def forward(self, bert_features, attention_mask, nrc_features):
        nrc_projected = self.nrc_projection(nrc_features)
        nrc_sequence = nrc_projected.unsqueeze(1).expand(-1, bert_features.size(1), -1)
        fused_features = torch.cat([bert_features, nrc_sequence], dim=2)
        mask = attention_mask.unsqueeze(-1).to(fused_features.dtype)
        fused_features = fused_features * mask
        pooled_outputs = []
        for filter_size, convolution in zip(self.filter_sizes, self.convolutions):
            conv_output = F.gelu(convolution(fused_features))
            active_windows = F.max_pool1d(attention_mask.unsqueeze(1).to(conv_output.dtype), kernel_size=filter_size, stride=1) > 0
            conv_output = conv_output.masked_fill(~active_windows, torch.finfo(conv_output.dtype).min)
            pooled = F.adaptive_max_pool1d(conv_output, output_size=1).squeeze(2)
            pooled_outputs.append(pooled)
        combined = torch.cat(pooled_outputs, dim=1)
        combined = self.layer_norm(combined)
        combined = self.dropout(combined)
        return self.classifier(combined)

# ==========================================
# 4. INITIALIZE MODELS
# ==========================================
print("Loading Tokenizer and IndoBERT...")
tokenizer = AutoTokenizer.from_pretrained("tokenizer")
bert_encoder = AutoModel.from_pretrained(config["model_name"]).to(DEVICE)
bert_encoder.eval()
for param in bert_encoder.parameters():
    param.requires_grad = False

print("Loading Custom CNN Model...")
model = IndoBERTBaseP1NRCFusionCNN2D(
    bert_hidden_size=bert_encoder.config.hidden_size,
    num_classes=len(LABELS),
    nrc_input_dim=len(NRC_FEATURE_COLUMNS),
    nrc_projection_dim=config["nrc_projection_dim"],
    filter_sizes=config["filter_sizes"],
    num_filters=config["num_filters"],
    dropout=config["dropout"],
    nrc_projection_dropout=config["nrc_projection_dropout"]
).to(DEVICE)

model.load_state_dict(torch.load("best_indobert_base_p2_nrc_early_fusion_cnn2d_optimized_v2.pt", map_location=DEVICE))
model.eval()
print("All models loaded successfully.")

# ==========================================
# 5. FLASK ROUTES
# ==========================================
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        teks = request.form.get("text_input", "")
        if not teks:
            return render_template("index.html", error="Teks tidak boleh kosong.")

        # 1. Preprocess
        teks_preprocessed = preprocessing_indobert(teks)
        
        # 2. Tokenize (Dynamic Batch Trimming)
        encoding = tokenizer(
            teks_preprocessed, add_special_tokens=True, max_length=MAX_LENGTH,
            padding="max_length", truncation=True, return_attention_mask=True, return_tensors="pt"
        )
        panjang_valid = max(int(encoding["attention_mask"].sum().item()), max(config["filter_sizes"]))
        input_ids = encoding["input_ids"][:, :panjang_valid].to(DEVICE)
        attention_mask = encoding["attention_mask"][:, :panjang_valid].to(DEVICE)

        # 3. NRC Features Extraction
        nrc_raw_features = ekstraksi_fitur_nrc(teks)
        nrc_feature_frame = pd.DataFrame([nrc_raw_features], columns=NRC_FEATURE_COLUMNS)
        nrc_scaled = nrc_scaler.transform(nrc_feature_frame).astype(np.float32)
        nrc_features = torch.tensor(nrc_scaled, dtype=torch.float32, device=DEVICE)

        # 4. Inference
        with torch.inference_mode():
            bert_output = bert_encoder(input_ids=input_ids, attention_mask=attention_mask)
            bert_features = bert_output.last_hidden_state.to(torch.float32)
            logits = model(bert_features=bert_features, attention_mask=attention_mask, nrc_features=nrc_features)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        # 5. Formatting Output
        hasil = [{"emosi": label, "probabilitas": float(prob)} for label, prob in zip(LABELS, probabilities)]
        hasil = sorted(hasil, key=lambda x: x["probabilitas"], reverse=True)
        prediksi_utama = hasil[0]["emosi"]

        return render_template(
            "index.html", 
            original_text=teks, 
            preprocessed=teks_preprocessed,
            prediction=prediksi_utama, 
            probabilities=hasil
        )

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=5000)
