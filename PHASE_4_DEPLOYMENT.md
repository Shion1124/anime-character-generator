# Phase 4: デプロイメント（UI + API + Cloud）実装ガイド

**対象フェーズ**: Phase 4 (本番環境展開)  
**推定期間**: 7-10日  
**技術スタック**: Streamlit (フロントエンド) + FastAPI (バックエンド) + Docker (コンテナ化) + GCP/Heroku (クラウド)  
**依存**: Phase 2A LoRA + Phase 2B LCM (オプション) + Phase 3 マルチモーダル  
**成果物**: 完全なProduction サービス (Web UI + REST API)

---

## 📖 背景：なぜデプロイメントが必要か？

### 問題: ローカルのみの制限

Phase 1-3 までは開発環境でのみ実行：

```
制限事項:
  ❌ 他のユーザーが使用不可
  ❌ Colab セッションに依存
  ❌ GPU インスタンス起動・停止の手間
  ❌ スケーラビリティなし
  ❌ REST API なし（他アプリからの連携不可）
```

### 解決策: Production-ready サービス

Phase 4 で実装する 3-層構成：

```
┌─────────────────────────────────────┐
│   Web UI (Streamlit)                │ ← ユーザーインタフェース
│   - 画像アップロード                 │
│   - プロンプト入力                   │
│   - リアルタイムプレビュー           │
└────────────────┬────────────────────┘
                 │ HTTP
                 ▼
┌─────────────────────────────────────┐
│   REST API (FastAPI)                │ ← ビジネスロジック
│   - /generate (T2I)                 │
│   - /img2img (I2I)                  │
│   - /controlnet (Pose/Edge)         │
│   - /inpaint (局所編集)              │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│   AI Model Layer                    │ ← 推論エンジン
│   - Stable Diffusion v1.5           │
│   - LoRA (Phase 2A)                 │
│   - LCM (Phase 2B)                  │
│   - ControlNet (Phase 3)            │
└─────────────────────────────────────┘

デプロイメント:
  開発: localhost:8000
  本番: GCP/Heroku (auto-scaling)
```

---

## 🎯 技術アーキテクチャ

### アーキテクチャ図

```
Internet
   ↑
   │ HTTPS
   ▼
[Load Balancer (GCP)]
   │
   ├─ [Instance 1] → [GPU] → [Model Cache]
   ├─ [Instance 2] → [GPU] → [Model Cache]
   └─ [Instance 3] → [GPU] → [Model Cache]
   
   キャッシング:
   - モデルは 1 回のみロード
   - 生成画像は Redis にキャッシュ
   - プロンプトの重複リクエストは即座に返却
```

### 技術選定理由

| 層 | ツール | 理由 |
|---|-------|-----|
| フロントエンド | Streamlit | Python ネイティブ、高速プロトタイピング |
| バックエンド | FastAPI | 非同期対応、自動ドキュメント |
| コンテナ | Docker | 環境統一、本番移行の容易さ |
| クラウド | GCP Compute Engine | GPU インスタンス安価、自動スケーリング対応 |
| 代替案 | Heroku | セットアップ簡易、スケーリング手動 |

---

## 🛠️ 実装ステップ

### Step 1: FastAPI バックエンド

**ファイル**: `api_server.py` を新規作成

```python
#!/usr/bin/env python3
"""
FastAPI Backend: アニメキャラ生成 REST API

エンドポイント:
  POST /generate        → テキスト→画像
  POST /img2img         → 画像→画像
  POST /controlnet      → ControlNet
  POST /inpaint         → 局所編集
  GET  /health          → ヘルスチェック
  GET  /models          → ロード済みモデル情報

使用例:
  pip install fastapi uvicorn python-multipart
  python api_server.py
  # http://localhost:8000/docs で Swagger UI
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import torch
import os
from pathlib import Path
from PIL import Image
import io
import base64
import time
from datetime import datetime
import json
import uvicorn
import logging

# ローカルモジュール
from character_generator import AnimeCharacterGenerator
from multimodal_pipeline import MultimodalPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== Pydantic Models ==========

class GenerateRequest(BaseModel):
    """T2I リクエスト"""
    prompt: str
    negative_prompt: Optional[str] = ""
    num_steps: Optional[int] = 20
    guidance_scale: Optional[float] = 7.5
    use_lcm: Optional[bool] = False

class Img2ImgRequest(BaseModel):
    """I2I リクエスト"""
    prompt: str
    negative_prompt: Optional[str] = ""
    strength: Optional[float] = 0.8
    num_steps: Optional[int] = 20
    guidance_scale: Optional[float] = 7.5

class ControlNetRequest(BaseModel):
    """ControlNet リクエスト"""
    prompt: str
    negative_prompt: Optional[str] = ""
    mode: Optional[str] = "pose"  # pose / edges / depth
    num_steps: Optional[int] = 20
    guidance_scale: Optional[float] = 7.5

class InpaintRequest(BaseModel):
    """Inpainting リクエスト"""
    prompt: str
    negative_prompt: Optional[str] = ""
    strength: Optional[float] = 0.8
    num_steps: Optional[int] = 30

# ========== FastAPI App ==========

app = FastAPI(
    title="Anime Character Generation API",
    version="1.0.0",
    description="Phase 1-3 統合アニメキャラ生成 API",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 有効化（フロントエンドからアクセス可能に）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== グローバル変数 ==========

generator = None
multimodal = None
generation_count = 0
start_time = datetime.now()

# ========== 初期化 ==========

@app.on_event("startup")
async def startup_event():
    """アプリケーション起動時"""
    
    global generator, multimodal
    
    logger.info("🚀 Initializing API Server")
    
    # GPU チェック
    if not torch.cuda.is_available():
        logger.warning("⚠️  CUDA not available, using CPU (slow)")
    else:
        logger.info(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    
    # モデルロード
    logger.info("📦 Loading character generator")
    generator = AnimeCharacterGenerator(
        use_lcm=False  # デフォルト: LCM なし（精度優先）
    )
    
    logger.info("📦 Loading multimodal pipeline")
    multimodal = MultimodalPipeline(
        lora_path="./lora_weights/anime-lora-final",
        use_lcm=False
    )
    
    logger.info("✅ API Server ready")

# ========== エンドポイント ==========

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    
    uptime = (datetime.now() - start_time).total_seconds()
    
    return {
        "status": "ok",
        "uptime_seconds": uptime,
        "gpu_available": torch.cuda.is_available(),
        "total_generations": generation_count,
        "memory_mb": {
            "reserved": torch.cuda.memory_reserved() / 1024 / 1024,
            "allocated": torch.cuda.memory_allocated() / 1024 / 1024
        }
    }

@app.get("/models")
async def get_models_info():
    """ロード済みモデル情報"""
    
    return {
        "base_model": "runwayml/stable-diffusion-v1-5",
        "lora": "./lora_weights/anime-lora-final",
        "features": [
            "text-to-image",
            "image-to-image",
            "controlnet-pose",
            "controlnet-edge",
            "inpainting"
        ],
        "gpu_memory_mb": {
            "total": torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 if torch.cuda.is_available() else 0
        }
    }

@app.post("/generate")
async def generate_image(request: GenerateRequest):
    """
    テキスト→画像 (T2I)
    
    例:
    ```
    curl -X POST http://localhost:8000/generate \
      -H "Content-Type: application/json" \
      -d '{
        "prompt": "beautiful anime girl",
        "num_steps": 20
      }'
    ```
    """
    
    global generation_count
    
    if not generator:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        logger.info(f"Generating: {request.prompt[:50]}")
        
        start = time.time()
        
        image = generator.generate_image(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.num_steps,
            guidance_scale=request.guidance_scale
        )
        
        elapsed = time.time() - start
        generation_count += 1
        
        # 画像を base64 エンコード
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        image_b64 = base64.b64encode(image_bytes.getvalue()).decode()
        
        return {
            "success": True,
            "image_base64": image_b64,
            "generation_time_s": elapsed,
            "prompt": request.prompt,
            "total_generations": generation_count
        }
    
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/img2img")
async def img2img(
    input_image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    strength: float = Form(0.8),
    num_steps: int = Form(20)
):
    """
    Image-to-Image 変換
    
    例:
    ```
    curl -X POST http://localhost:8000/img2img \
      -F "input_image=@sketch.png" \
      -F "prompt=anime girl" \
      -F "strength=0.8"
    ```
    """
    
    global generation_count
    
    if not multimodal:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # 画像読み込み
        image_data = await input_image.read()
        input_img = Image.open(io.BytesIO(image_data))
        
        logger.info(f"I2I: {prompt[:50]}")
        
        start = time.time()
        
        output = multimodal.image_to_image(
            input_image=input_img,
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=strength,
            num_inference_steps=num_steps
        )
        
        elapsed = time.time() - start
        generation_count += 1
        
        # Base64 エンコード
        output_bytes = io.BytesIO()
        output.save(output_bytes, format="PNG")
        output_b64 = base64.b64encode(output_bytes.getvalue()).decode()
        
        return {
            "success": True,
            "image_base64": output_b64,
            "generation_time_s": elapsed,
            "prompt": prompt,
            "strength": strength
        }
    
    except Exception as e:
        logger.error(f"I2I error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/inpaint")
async def inpaint(
    input_image: UploadFile = File(...),
    mask_image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form("")
):
    """
    局所編集 (Inpainting)
    
    例:
    ```
    curl -X POST http://localhost:8000/inpaint \
      -F "input_image=@character.png" \
      -F "mask_image=@mask.png" \
      -F "prompt=blue hair"
    ```
    """
    
    global generation_count
    
    if not multimodal:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # 画像読み込み
        input_data = await input_image.read()
        mask_data = await mask_image.read()
        
        input_img = Image.open(io.BytesIO(input_data))
        mask_img = Image.open(io.BytesIO(mask_data))
        
        logger.info(f"Inpainting: {prompt[:50]}")
        
        start = time.time()
        
        output = multimodal.inpaint(
            input_image=input_img,
            mask_image=mask_img,
            prompt=prompt,
            negative_prompt=negative_prompt
        )
        
        elapsed = time.time() - start
        generation_count += 1
        
        # Base64 エンコード
        output_bytes = io.BytesIO()
        output.save(output_bytes, format="PNG")
        output_b64 = base64.b64encode(output_bytes.getvalue()).decode()
        
        return {
            "success": True,
            "image_base64": output_b64,
            "generation_time_s": elapsed,
            "prompt": prompt
        }
    
    except Exception as e:
        logger.error(f"Inpainting error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ========== メイン ==========

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
```

### Step 2: Streamlit フロントエンド

**ファイル**: `streamlit_app.py` を新規作成

```python
#!/usr/bin/env python3
"""
Streamlit Web UI: アニメキャラ生成フロントエンド

実行:
    streamlit run streamlit_app.py --server.port=8501
"""

import streamlit as st
from streamlit_option_menu import option_menu
import requests
import base64
from PIL import Image
import io
import time

# ページ設定
st.set_page_config(
    page_title="Anime Character Generator",
    page_icon="✨",
    layout="wide"
)

# API エンドポイント
API_URL = "http://localhost:8000"

# ========== Sidebar ==========

st.sidebar.title("🎨 Anime Generator")
st.sidebar.markdown("---")

# メニュー
with st.sidebar:
    selected = option_menu(
        menu_title="メニュー",
        options=["T2I（テキスト→画像）", "I2I（画像→画像）", "Inpainting（局所編集）", "バッチ処理", "API ドキュメント"],
        icons=["sparkles", "image", "pencil-square", "files", "book"],
        menu_icon="cast"
    )

st.sidebar.markdown("---")

# ヘルスチェック
try:
    health = requests.get(f"{API_URL}/health", timeout=2).json()
    st.sidebar.metric("🟢 Status", "Online")
    st.sidebar.metric("生成数", health["total_generations"])
except:
    st.sidebar.metric("🔴 Status", "Offline")
    st.error("⚠️  API サーバーに接続できません")

# ========== ページ実装 ==========

if selected == "T2I（テキスト→画像）":
    st.title("✨ テキスト→画像 生成")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        prompt = st.text_area(
            "プロンプト",
            value="beautiful anime girl, long hair, masterpiece",
            height=100
        )
        negative_prompt = st.text_area(
            "ネガティブプロンプト",
            value="low quality, blurry",
            height=50
        )
    
    with col2:
        num_steps = st.slider("ステップ数", 10, 50, 20)
        guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5)
        use_lcm = st.checkbox("LCM 使用（高速化）")
    
    if st.button("🎨 生成", use_container_width=True):
        with st.spinner("生成中..."):
            try:
                start = time.time()
                
                response = requests.post(
                    f"{API_URL}/generate",
                    json={
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "num_steps": num_steps,
                        "guidance_scale": guidance_scale,
                        "use_lcm": use_lcm
                    },
                    timeout=300
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # 画像表示
                    image_data = base64.b64decode(data["image_base64"])
                    image = Image.open(io.BytesIO(image_data))
                    
                    elapsed = time.time() - start
                    
                    st.image(image, use_column_width=True)
                    st.success(f"✅ 生成完了 ({elapsed:.1f}s)")
                    st.json({
                        "prompt": data["prompt"],
                        "generation_time_s": f"{data['generation_time_s']:.2f}",
                        "total_generations": data["total_generations"]
                    })
                else:
                    st.error(f"❌ エラー: {response.json()['detail']}")
            
            except Exception as e:
                st.error(f"❌ エラー: {e}")


elif selected == "I2I（画像→画像）":
    st.title("🖼️  Image-to-Image 変換")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("入力画像")
        uploaded_file = st.file_uploader("画像をアップロード", type=["png", "jpg", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("設定")
        prompt = st.text_area("プロンプト", value="anime character, masterpiece")
        strength = st.slider("変更度", 0.0, 1.0, 0.8)
        num_steps = st.slider("ステップ数", 10, 50, 20)
    
    if st.button("✨ 変換", use_container_width=True) and uploaded_file:
        with st.spinner("処理中..."):
            try:
                files = {
                    "input_image": uploaded_file.getvalue(),
                }
                data = {
                    "prompt": prompt,
                    "strength": strength,
                    "num_steps": num_steps
                }
                
                response = requests.post(
                    f"{API_URL}/img2img",
                    files=files,
                    data=data,
                    timeout=300
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    output_data = base64.b64decode(result["image_base64"])
                    output_image = Image.open(io.BytesIO(output_data))
                    st.image(output_image, use_column_width=True)
                    st.success(f"✅ 完了 ({result['generation_time_s']:.1f}s)")
                else:
                    st.error(f"❌ エラー: {response.json()['detail']}")
            
            except Exception as e:
                st.error(f"❌ エラー: {e}")


elif selected == "Inpainting（局所編集）":
    st.title("✏️  局所編集 (Inpainting)")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.subheader("元画像")
        image_file = st.file_uploader("画像", type=["png", "jpg"], key="base_img")
        if image_file:
            st.image(image_file, use_column_width=True)
    
    with col2:
        st.subheader("マスク")
        mask_file = st.file_uploader("マスク画像（白=編集）", type=["png", "jpg"], key="mask_img")
        if mask_file:
            st.image(mask_file, use_column_width=True)
    
    with col3:
        st.subheader("設定")
        prompt = st.text_area("編集内容", value="blue hair")
    
    if st.button("🎨 適用", use_container_width=True) and image_file and mask_file:
        with st.spinner("処理中..."):
            try:
                response = requests.post(
                    f"{API_URL}/inpaint",
                    files={
                        "input_image": image_file.getvalue(),
                        "mask_image": mask_file.getvalue()
                    },
                    data={"prompt": prompt},
                    timeout=300
                )
                
                if response.status_code == 200:
                    result = response.json()
                    output_data = base64.b64decode(result["image_base64"])
                    output_image = Image.open(io.BytesIO(output_data))
                    st.image(output_image, use_column_width=True)
                    st.success(f"✅ 完了 ({result['generation_time_s']:.1f}s)")
                else:
                    st.error(f"❌ エラー")
            
            except Exception as e:
                st.error(f"❌ エラー: {e}")


elif selected == "バッチ処理":
    st.title("📦 バッチ処理")
    st.info("複数画像の一括生成はバックエンド経由で実行してください")
    
    st.code("""
# Python スクリプト例
import requests

for i in range(10):
    response = requests.post(f"{API_URL}/generate", json={
        "prompt": "anime girl"
    })
    # 結果処理
    """, language="python")


elif selected == "API ドキュメント":
    st.title("📚 API ドキュメント")
    
    st.markdown("""
    ### エンドポイント
    
    #### POST /generate
    テキスト→画像 生成
    
    **リクエスト:**
    ```json
    {
      "prompt": "beautiful anime girl",
      "negative_prompt": "",
      "num_steps": 20,
      "guidance_scale": 7.5,
      "use_lcm": false
    }
    ```
    
    **レスポンス:**
    ```json
    {
      "success": true,
      "image_base64": "...",
      "generation_time_s": 5.2,
      "total_generations": 42
    }
    ```
    
    ---
    
    #### POST /img2img
    Image-to-Image 変換
    
    **パラメータ:**
    - `input_image`: 画像ファイル
    - `prompt`: プロンプト
    - `strength`: 変更度 (0-1)
    - `num_steps`: ステップ数
    
    ---
    
    #### POST /inpaint
    局所編集
    
    **パラメータ:**
    - `input_image`: 元画像
    - `mask_image`: マスク画像
    - `prompt`: 編集内容
    
    ---
    
    #### GET /health
    ヘルスチェック
    
    #### GET /models
    ロード済みモデル情報
    
    ---
    
    ### 完全ドキュメント
    
    http://localhost:8000/docs (Swagger UI)
    """)
```

### Step 3: Docker & Deployment

```dockerfile
# Dockerfile

FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app

# Python インストール
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# 依存パッケージ
COPY requirements_deploy.txt .
RUN pip install -q -r requirements_deploy.txt

# アプリケーションコピー
COPY api_server.py .
COPY streamlit_app.py .
COPY character_generator.py .
COPY multimodal_pipeline.py .
COPY lora_weights/ ./lora_weights/

# ポート
EXPOSE 8000 8501

# スタートスクリプト
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

CMD ["./docker-entrypoint.sh"]
```

```bash
# docker-entrypoint.sh

#!/bin/bash

# API サーバー起動（バックグラウンド）
python api_server.py &

# Streamlit UI 起動
streamlit run streamlit_app.py \
  --server.port=8501 \
  --server.address=0.0.0.0 \
  --client.toolbarPosition=bottom
```

```yaml
# docker-compose.yml

version: '3.8'

services:
  anime-generator:
    build: .
    ports:
      - "8000:8000"  # API
      - "8501:8501"  # Streamlit UI
    environment:
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ./outputs:/app/outputs
      - ./lora_weights:/app/lora_weights
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## 📊 デプロイメント戦略

### 開発環境

```bash
# ローカル実行
python api_server.py  # ターミナル 1
streamlit run streamlit_app.py --server.port=8501  # ターミナル 2

# アクセス
# API: http://localhost:8000/docs
# UI: http://localhost:8501
```

### 本番環境 (GCP)

```bash
# Container Registry へプッシュ
docker build -t gcr.io/PROJECT_ID/anime-gen:latest .
docker push gcr.io/PROJECT_ID/anime-gen:latest

# Cloud Run へデプロイ
gcloud run deploy anime-generator \
  --image gcr.io/PROJECT_ID/anime-gen:latest \
  --platform managed \
  --region us-central1 \
  --memory 16Gi \
  --cpu 4 \
  --gpu 1  # GPU T4
  --timeout 600
```

### 本番環境 (Heroku)

```bash
# Heroku CLI
heroku login
heroku create anime-character-generator
heroku config:set GPU_MEMORY=0.9

# デプロイ
git push heroku main

# 確認
heroku logs --tail
```

---

## ✅ 完了チェックリスト

フェーズ 4 実装の完了基準：

### Backend (FastAPI)
- [ ] `api_server.py` 実装完了
- [ ] 全 5 エンドポイント動作確認
- [ ] エラーハンドリング実装
- [ ] スワッガー UI テスト (`/docs`)
- [ ] ローカル動作確認

### Frontend (Streamlit)
- [ ] `streamlit_app.py` 実装完了
- [ ] 5 ページメニュー実装
- [ ] 画像アップロード機能確認
- [ ] T2I, I2I, Inpainting テスト実行
- [ ] UI レスポンシブ確認

### Docker & Cloud
- [ ] `Dockerfile` 作成完了
- [ ] `docker-compose.yml` テスト
- [ ] ローカル Docker 実行確認
- [ ] GCP Deploy テスト
- [ ] Heroku Deploy テスト

### Integration
- [ ] API ↔ UI 連携確認
- [ ] 画像ダウンロード機能確認
- [ ] マルチリクエスト同時処理テスト
- [ ] メモリリーク確認

### Documentation
- [ ] デプロイメント手順書作成
- [ ] API 仕様書作成
- [ ] トラブルシューティングガイド
- [ ] ブログ記事「Production 環境へのデプロイ」

---

## 🎓 Phase 1-4 完了時の到達点

```
✅ Phase 1: LLM × Prompt Optimization
   - RobustPromptGenerator 実装
   - Anthropic Claude API 統合
   - プロンプトキャッシング

✅ Phase 2A: LoRA Fine-tuning
   - 300 枚の anime データセット
   - 20 エポック学習完了
   - anime-lora-final/ 出力

✅ Phase 2B: LCM Distillation
   - 4-step LCM モデル生成
   - 推論 12x 高速化確認

✅ Phase 3: Multimodal Operations
   - Image-to-Image パイプライン
   - ControlNet (Pose/Edges)
   - Inpainting 局所編集

✅ Phase 4: Production Deployment
   - FastAPI バックエンド
   - Streamlit フロントエンド
   - Docker コンテナ化
   - GCP/Heroku 本番展開

結果: 企業レベルの AI サービス完成 🚀
```

---

**プロジェクト完了おめでとうございます！** 🎉

次は → [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md) で全体進捗を確認

