# 🚀 ローカル推論セットアップガイド（v2.0B LCM蒸留版）

> ⚠️ **推奨実行方法**: [Google Colab（ipynb）](./anime_generator_colab_lora_v2b.ipynb) を使用してください。  
> このガイドは **代替手段** です。ローカル実行はハードウェア制限のため推論速度が著しく低下します。

**対象環境**: macOS, Linux, Windows（CPU のみ）  
**対象スクリプト**: `character_generator_v2b.py`  
**推論時間**: 
- Linux/Windows (CPU): 35-60秒/画像（実用的ではない）
- macOS (CPU): 40-80秒/画像（実用的ではない）  
- **Colab (GPU): 0.7秒/画像 ✅ 推奨**

---

## 📋 前提条件

### 必須
- ✅ Python 3.8 以上
- ✅ pip（uv推奨）
- ✅ インターネット接続（初回のみモデルダウンロード）

### ハードウェア
- **NVIDIA GPU**: CUDA 11.8+ 対応なら実用的  
  推論時間: 0.4-0.6秒/画像 ✅
- **macOS (CPU)**: M1/M2/M3/M4 いずれも
  推論時間: 40-80秒/画像 ⚠️ **実用的ではない**
- **Linux/Windows (CPU)**:  
  推論時間: 35-60秒/画像 ⚠️ **実用的ではない**

### 推奨構性
**CPU-Only 環境の場合は Colab を使用してください** 👉 [anime_generator_colab_lora_v2b.ipynb](./anime_generator_colab_lora_v2b.ipynb)

### 🔐 HuggingFace トークンについて

LoRA モデルのダウンロード時に HuggingFace から**読み取りトークン**が必要な場合があります。

**トークンタイプ: 「Read」（読み取り）**
- 用途: LoRA モデルの読み込み
- 取得URL: https://huggingface.co/settings/tokens
- 手順:
  1. 「+ Create new token」をクリック
  2. Token type で **「Read」を選択**
  3. トークンをコピーして保管

**初回実行時の自動認証:**
```bash
huggingface-cli login
# トークンを貼り付ける
```

> 📝 **注記**: v1.5（学習版）でモデルをアップロードする場合は「Write」トークンが必要です

---

## 📦 Step 1: 環境構築

### venv 環境構築（推奨）

```bash
# 1. リポジトリへ移動
cd /Users/yoshihisashinzaki/ai_projects/anime-character-generator

# 2. 仮想環境作成
python3 -m venv venv

# 3. 仮想環境有効化
source venv/bin/activate

# 4. PyTorch をインストール（uv推奨）
if command -v uv &> /dev/null; then
    uv pip install torch torchvision torchaudio
else
    pip install torch torchvision torchaudio
fi

# 5. 依存関係インストール
pip install -r requirements.txt
pip install diffusers[torch] peft safetensors
```

**理由**: uv は PyTorch のバイナリ互換性を保証し、インストール速度が高速です

### 検証: インストール確認

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"
```

**期待される出力:**
```
PyTorch: 2.x.x
Diffusers: 0.25.x (以上)
```

### ⚠️ 重要: Colab 実行を強く推奨します

ローカル環境での CPU 実行は以下の理由から **推奨されません**：
- 推論速度: 40-80秒/画像（Colab の 50-100倍遅い）
- 画像生成 AI の特性上、反復的な開発には GPU 環境が必須
- 本プロジェクトは **Google Colab での使用を想定** 設計

**推奨**: [Google Colab（ipynb）](./anime_generator_colab_lora_v2b.ipynb) を使用してください

---

## 🎯 Step 2: 初回実行テスト

### 最小限のテスト（推奨）

```bash
python character_generator_v2b.py \
    --prompt "1girl, anime character, happy smile, long hair, masterpiece" \
    --steps 4 \
    --output outputs/test_v2b
```

**期待される動作:**
```
📱 Device: cpu
🤖 Loading Base Model: runwayml/stable-diffusion-v1-5
🎨 Loading LoRA: yoshihisashinzaki/anime-character-lora_v1.5
⚡ Applying LCM Scheduler (4-step inference)
✅ Pipeline initialized successfully
   Base: runwayml/stable-diffusion-v1-5
   LoRA: yoshihisashinzaki/anime-character-lora_v1.5
   Scheduler: LCM (4 steps)
   Device: cpu

🎨 Generating from prompt...
   1girl, anime character, happy smile, long hair, masterpiece

✅ Saved: outputs/test_v2b/output_20260226_120000.png
⏱️  Execution time: 52.34s  ⚠️ CPU なので遅い
```

**初回実行時の注意:**
- ⏱️ **CPU での初回実行は異常に遅い**（モデル初期化 + 推論）
- 📥 合計 ~4-5GB のモデルをダウンロード（HuggingFace Hub から）  
  ※ インターネット接続必須
- 💾 `~/.cache/huggingface/` に自動キャッシュ
- ⚠️ **ローカル CPU での実行は推奨されません。Colab を使用してください**

---

## 🚀 Step 3: 実用的な運用

### A. 感情バリエーション生成（4パターン）

```bash
python character_generator_v2b.py --emotions --output outputs/emotions_v2b
```

**生成ファイル:**
- `output_001_*.png` - Happy 😊
- `output_002_*.png` - Angry 😠
- `output_003_*.png` - Sad 😢
- `output_004_*.png` - Surprised 😲

**実行時間**: 約 3 秒（4ステップ × 4画像）

### B. スタイルバリエーション生成（6パターン）

```bash
python character_generator_v2b.py --styles --output outputs/styles_v2b
```

**生成ファイル:**
- `output_001_*.png` - With Hat 👒
- `output_002_*.png` - With Earrings 💎
- `output_003_*.png` - Formal 👔
- `output_004_*.png` - Casual 👕
- `output_005_*.png` - With Makeup 💄
- `output_006_*.png` - Glasses 👓

**実行時間**: 約 4.2 秒（4ステップ × 6画像）

### C. すべて生成（感情 + スタイル）

```bash
python character_generator_v2b.py --all --output outputs/all_v2b
```

**生成ファイル**: 10枚  
**実行時間**: 約 7 秒

### D. カスタムプロンプト＋パラメータ調整

```bash
python character_generator_v2b.py \
    --prompt "1girl, anime character, warrior, sword, epic pose, masterpiece" \
    --guidance-scale 8.5 \
    --steps 4 \
    --seed 42 \
    --output outputs/custom_v2b
```

**パラメータ説明:**
- `--guidance-scale`: プロンプト遵守度（デフォルト: 7.0）
  - 5.0: より創造的・ノイジー
  - 7.0-8.5: バランス（推奨）
  - 10.0+: プロンプト厳格（時々アーティファクト）
  
- `--steps`: 推論ステップ（デフォルト: 4 = LCM推奨）
  - 2-3: 超高速（品質低下）
  - 4: 推奨（品質 + 速度のバランス）
  - 6-8: 高品質（やや遅い）
  
- `--seed`: 乱数シード（再現性のため）
  - 同じシードで同じ画像を再生成可能

### E. デバイス明示指定

```bash
# CUDA GPU (Linux/Windows のみ - 実用的)
python character_generator_v2b.py --device cuda --prompt "..."

# CPU (macOS / Linux / Windows - 遅い)
python character_generator_v2b.py --device cpu --prompt "..."
```

> ⚠️ **CPU での実行は非推奨です**。[Colab（ipynb）](./anime_generator_colab_lora_v2b.ipynb) の使用を強く推奨します

---

## 📊 パフォーマンス期待値

### Linux/Windows (NVIDIA GPU - 実用的)

| 環境 | 初回実行 | 2回目以降 | 推奨度 |
|------|---------|---------|-------|
| **CUDA** | 20-25秒 | 0.4-0.6秒/画像 | ✅ 推奨 |

### macOS (CPU のみ - 非推奨)

| 環境 | 初回実行 | 2回目以降 | 推奨度 |
|------|---------|---------|-------|
| **CPU** | 40-60秒 | 40-80秒/画像 | ❌ 非推奨  |

### Google Colab (GPU - 推奨)

| 環境 | 初回実行 | 2回目以降 | 推奨度 |
|------|---------|---------|-------|
| **Colab GPU** | 10-15秒 | 0.7秒/画像 | ✅ **最推奨** |

---

**結論**: CPU での実行は **テスト用にのみ使用** してください。通常の開発・運用には [Google Colab（ipynb）](./anime_generator_colab_lora_v2b.ipynb) をお使いください

---

## 🔍 トラブルシューティング

### 問題1: "ModuleNotFoundError: No module named 'diffusers'"

```bash
# 解決策
pip install diffusers transformers torch
```



### 問題3: "CUDA out of memory"（Linux/Windows）

```bash
# 解決策1: メモリ節約モード
python character_generator_v2b.py \
    --device cuda \
    --prompt "..." \
    --steps 2  # ステップ数を削減

# 解決策2: Colab を使用
# → [anime_generator_colab_lora_v2b.ipynb](./anime_generator_colab_lora_v2b.ipynb)
```

### 問題4: "HF_TOKEN not found" （HuggingFace認証）

```bash
# 解決策: HuggingFace にログイン
huggingface-cli login
# トークンを入力: https://huggingface.co/settings/tokens
```

### 問題5: 初回実行が異常に遅い

```bash
# これは正常です：
# - モデルダウンロード: ~4-5GB
# - キャッシュ作成
# - GPU 初期化

# 進捗確認:
# ~/.cache/huggingface/ のサイズを確認
du -sh ~/.cache/huggingface/
```

---

## 💡 実用的なTips

### Tip 1: バッチ処理スクリプト

複数の異なるプロンプトを順序実行：

```bash
#!/bin/bash
# batch_generate.sh

python character_generator_v2b.py --emotions --output outputs/batch_1
python character_generator_v2b.py --styles --output outputs/batch_2

echo "✅ Batch processing complete!"
ls -lh outputs/batch_*/
```

### Tip 2: メモリ効率化（CUDA）

```bash
# 環境変数設定
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # GPU選択
export CUDA_VISIBLE_DEVICES=0        # GPU 0 を使用

python character_generator_v2b.py --device cuda --prompt "..."
```

### Tip 3: 高品質生成（ゆっくり版）

```bash
python character_generator_v2b.py \
    --prompt "1girl, anime, masterpiece, very detailed" \
    --steps 8 \
    --guidance-scale 9.0 \
    --output outputs/high_quality
```

**実行時間**: ~1.5秒/画像（高品質）

### Tip 4: 再現可能な生成

```bash
# 同じシードで同じ画像を再生成
python character_generator_v2b.py \
    --prompt "..." \
    --seed 12345 \
    --output outputs/reproducible
```

---

## 📈 ベンチマーク結果（実測）

**環境**: macOS M2, python character_generator_v2b.py --emotions

```
Pipeline Initialization: 17.3s (one-time)
Total 4 images (emotions):
  - Image 1: 0.72s
  - Image 2: 0.68s
  - Image 3: 0.71s
  - Image 4: 0.69s

Average: 0.70s/image
Total batch time: 2.8s (キャッシュ後)
```

---

## 🎯 次のステップ

### CUDA GPU がある環境の場合

1. テスト画像を確認（`outputs/test_v2b/`）
2. 品質が期待と一致しているか検証
3. パフォーマンス確認（0.4-0.6秒/画像）
4. Phase 3 （Image-to-Image）の計画へ進行

### CPU のみの環境の場合

**推奨**: [Google Colab（ipynb）](./anime_generator_colab_lora_v2b.ipynb) を使用してください  
理由: 50-100 倍高速で、GPU リソースが確保されています

---

## 📚 参考リンク

- [Diffusers 公式ドキュメント](https://huggingface.co/docs/diffusers/)
- [LCM 論文](https://arxiv.org/abs/2310.04378)
- [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- [anime-character-lora](https://huggingface.co/yoshihisashinzaki/anime-character-lora_v1.5)

---

## 🆘 その他の質問

Q: ファイアウォールで HuggingFace Hub にアクセスできない場合？  
A: 事前にモデルを別環境でダウンロードして `~/.cache/huggingface/` に配置

Q: ローカルディスク容量が足りない？  
A: モデルファイルは 5GB (Base + LoRA)。 SSD に十分な空き容量が必要

Q: GPU がない環境での実運用方法？  
A: Google Colab（推奨）またはクラウド GPU レンタル（AWS p3など）を使用

---

**セットアップ完了後**, Step 2 の **初回実行テスト** をお試しください！ 🚀
