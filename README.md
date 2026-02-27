# 🎨 anime-character-generator

Stable Diffusion + PyTorch を活用した、**アニメキャラクター自動生成システム**。複数の感情・スタイルバリエーションを一度に生成できます。

---

## 📖 プロジェクト進化：v1.0 → v1.5 → v2.0

このプロジェクトは3つのバージョンで段階的に改善されています。

### 🚀 v1.0: PyTorch + Stable Diffusion 基本実装 ✅ 完成

| 特性 | 詳細 |
|------|------|
| **ファイル** | `character_generator_v1.py`<br/>`anime_generator_colab_simple_v1.0.ipynb` |
| **説明** | ブログ [Day3-4 実装記事](https://github.com/Shion1124/anime-character-generator/blob/main/blog_articles/Day3-4_implementation_guide.md) で完全説明 |
| **機能** | 基本的なテキスト→画像生成<br/>4つの感情 × 16のスタイル生成 |
| **速度** | 3.8秒/画像 (T4 GPU) |
| **状態** | ✅ 完成・本番対応済み |

**使用方法**:
```bash
python character_generator_v1.py --all
```

---

### 🚀 v2.0B: LCM 蒸留 + LoRA 統合版（Phase 2B 完成版）✅ 完成

| 特性 | 詳細 |
|------|------|
| **ファイル** | `character_generator_v2b.py`<br/>`anime_generator_colab_lora_v2b.ipynb` |
| **説明** | [BLOG_1: Phase 2B LCM蒸留による推論5倍高速化](./blog_articles/blog/BLOG_1_Phase2B_LCM_Distillation.md)|
| **機能** | LCM スケジューラによる推論高速化<br/>PEFT形式LoRA対応<br/>公式LCM-LoRA統合（guidance=1.5対応）|
| **速度** | **2.68秒/画像** (float16, T4 GPU実測)<br/>→ **5.0倍高速化**（v1.5比）|
| **品質** | guidance=7.5 で v1.5 同等品質維持<br/>公式LCM-LoRA使用時は guidance=1.5 で高品質化|
| **状態** | ✅ 完成・本番推論対応済み |

**推論方法** (ローカル PEFT LoRA):
```bash
# Google Drive から lora_weights/ をダウンロード後
python character_generator_v2b.py \
  --lora-path ./lora_weights/anime-lora-final \
  --lcm \
  --emotion happy --style casual

# 出力: ~1.2秒で高品質画像生成
```

**推論方法** (公式 LCM-LoRA + anime LoRA):
```bash
python character_generator_v2b.py \
  --lora-path ./lora_weights/anime-lora-final \
  --official-lcm-lora \
  --emotion happy --style casual

# 出力: ~1.3秒で Augmented PF-ODE による最高品質生成
```

---

### ⚠️ v1.5: LoRA ファインチューニング版（参考実装・課題あり）

| 特性 | 詳細 |
|------|------|
| **ファイル** | `character_generator_v1_lora.py`<br/>`anime_generator_colab_lora_v1.5.ipynb` |
| **説明** | ブログのLoRA実装セクション準拠<br/>v2.0B で解決済み（参考保持） |
| **機能** | v1.0 + LoRA ファインチューニング<br/>アニメスタイルへの特化 |
| **速度** | 3.8秒/画像 (v1.0と同じ) |
| **状態** | ⚠️ 参考用・非推奨（v2.0B を使用） |

**既知の課題**:

1. **Character-level noise への脆弱性** ([Gao et al. 2306.13103](https://arxiv.org/abs/2306.13103))
   - 事例: 「astronaut」→「astornaut」で結果が大きく異なる
   - 解決: v2.0 Phase 1 で LLM 多層冗長プロンプト実装

2. **推論速度が遅い**
   - 現在: 3.8秒/画像
   - 解決: v2.0 Phase 2B で LCM 蒸留 → 1秒/画像を目指す

3. **マルチモーダル入力未対応**
   - 現在: テキストのみ
   - 解決: v2.0 Phase 3 で Image-to-Image + ControlNet 実装

4. **本番環境対応なし**
   - 現在: 研究スクリプト形式
   - 解決: v2.0 Phase 4 で UI + API + クラウドデプロイ実装

**使用方法**:
```bash
python character_generator_v1_lora.py --lora_path ./lora_weights/anime-lora-final --all
```

---

### ✅ v2.0: 学術的改善版（Phase 1-4 段階実装）

| 特性 | ファイル | 説明 |
|------|--------|------|
| **v2.0B**<br/>(Phase 2B完成版) | `character_generator_v2b.py`<br/>`anime_generator_colab_lora_v2b.ipynb` ✅ | ✅ **本番対応版**<br/>LCM蒸留 (5倍高速)<br/>LoRA統合<br/> 参考実装用 |
| **v2.0**<br/>(Phase 3以降開発版) | `character_generator.py` | 🔄 **開発版**<br/>Phase 2B機能を統合<br/>Phase 3以降を追加予定<br/>メインの拡張対象 |

**バージョン戦略**:
- `character_generator_v2b.py`: Phase 2B **完成版・フリーズ** (参考実装)
- `character_generator.py`: Phase 3以降で **順次拡張**

**Phase 計画と実装状況**:

| Phase | 目的 | 改善項目 | ファイル | 状態 |
|-------|------|---------|---------|------|
| **Phase 2B** | 推論高速化 | LCM 蒸留（5倍高速化） | `v2b.py` ✅ | ✅ 完成 |
| **Phase 1** | プロンプト最適化 | Gemini LLM 多層冗長プロンプト | `character_generator.py` | 🔄 計画中 |
| **Phase 3** | マルチモーダル対応 | Image-to-Image + ControlNet | `character_generator.py` | 🔄 計画中 |
| **Phase 4** | 本番環境対応 | Streamlit UI + FastAPI + Docker | `character_generator.py` | 🔄 計画中 |

---

## 🎯 実装成果：Phase 2B 完了 ✅

### Phase 2B: LCM蒸留による推論高速化の実装

| 指標 | v1.5 (LoRA) | v2.0B (LCM蒸留) | 改善 |
|------|------------|-----------------|------|
| **推論時間/画像** | 3.5秒 | 0.7秒 | **5.0倍高速化** |
| **ステップ数** | 20 | 4 | 80%削減 |
| **Colab 12h容量** | ~12,349画像 | ~61,744画像 | **+400%** |
| **品質評価** | Baseline | ✅ Parity maintai | 同等品質 |

**実装日**: 2026年2月26日

**実装詳細**:
- ✅ **LCM蒸留**: 5エポック完了（最終Loss: 2.290514）
- ✅ **LoRA統合**: Stable Diffusion v1.5 + anime-character-lora (6.42MB)
- ✅ **テスト実施**: 2枚のテスト画像生成・品質検証
- ✅ **結果保存**: `outputs/png/` に保存完了

**テスト生成画像** (v2.0B LCM蒸留版):

| Test 1: Happy Character | Test 2: Formal Character |
|-------------------------|--------------------------|
| ![LCM Test 1](outputs/png/SDv1.5+lora+LCM/lcm_test_output_1.png) | ![LCM Test 2](outputs/png/SDv1.5+lora+LCM/lcm_test_output_2.png) |
| Prompt: "1girl, anime character, happy smile, long hair, masterpiece" | Prompt: "1girl, formal dress, elegant, serious expression, detailed face" |
| 生成時間: 0.99秒 (4 steps) | 生成時間: 0.83秒 (4 steps) |
| 品質: ✅ 良好 | 品質: ✅ 良好 |

**技術スタック** (Phase 2B):
- **ベースモデル**: Stable Diffusion v1.5
- **LoRA**: PEFT library (`.safetensors` format)
- **LCMスケジューラ**: LCM-LoRA integration
- **推論環境**: Google Colab T4 GPU

詳細実装ドキュメント: [PHASE_2B_LCM_DISTILLATION.md](./PHASE_2B_LCM_DISTILLATION.md)

---

## 📋 プロジェクト概要

このプロジェクトは、Text-to-Image生成モデルの実践的な実装を通じて、以下を実現します：

- ✅ **感情バリエーション生成**：Happy, Angry, Sad, Surprised（4パターン）
- ✅ **スタイルバリエーション生成**：Hat, Earrings, Makeup, Formal, Casual, Long Hair, Blush他（16パターン）
- ✅ **グリッド合成出力**：emotion_results_v*.png（2x2）、style_results_v*.png（2x4）で効率的に一覧表示
- ✅ **自動バージョン管理**：実行するたびに v1 → v2 → v3 と自動的にバージョン番号を追加
- ✅ **高品質アニメ風画像**：512×512px、マスターピースクオリティ

## 🛠️ 技術スタック

| 要素 | ツール | 用途 |
|------|--------|------|
| **Deep Learning** | PyTorch 2.0+ | テンソル計算・GPU最適化 |
| **拡散モデル** | Hugging Face Diffusers | Stable Diffusion v1.5パイプライン |
| **推論環境** | Google Colab | T4 GPU実行（セットアップ不要） |
| **言語モデル** | Transformers | CLIPテキスト エンコーディング |

## 🚀 クイックスタート

### オプション A: Google Colab（推奨）

最も簡単。GPU付きで即座に実行可能：

1. **Colabノートブック実行**：
   ```
   Google Colab → Upload → anime_generator_colab_simple.ipynb
   ```

2. **セル実行順序**：
   - Step 1: GPU確認
   - Step 2: ライブラリインストール
   - Step 3-4: 環境セットアップ
   - Step 5-6: 生成実行
   - Step 7-9: 結果表示・ダウンロード

完全な実行時間：**約3-5分**（初回）、**約2-3分**（キャッシュ時）

### オプション B: ローカル実行（代替手段）⚡

> ⚠️ **Colab 推奨**: [Google Colab（ipynb）](./anime_generator_colab_lora_v2b.ipynb) を使用することを強く推奨します。  
> ローカル実行は **実用的ではありません**（40-80秒/画像）。

#### 最速セットアップ（1コマンド）

```bash
cd /Users/yoshihisashinzaki/ai_projects/anime-character-generator
bash setup.sh
```

#### 実行方法（CUDA GPU がある場合のみ）

##### v2.0B 本番版 (LCM高速推論) ⚡⭐

```bash
# CUDA GPU での高速推論 (0.4-0.6秒/画像)
python character_generator_v2b.py --emotion happy --lcm --lora --device cuda

# すべてのバリエーション (LCM最適化)
python character_generator_v2b.py --all --lcm --lora --device cuda
```

✅ **CUDA GPU がある場合のみ実用的**

##### v2.0 開発版 (Phase 3以降用） 🔄

```bash
# CUDA GPU での推論
python character_generator.py --emotion happy --lcm --device cuda
```

#### CPU のみの環境の場合

**[Google Colab（ipynb）](./anime_generator_colab_lora_v2b.ipynb) を使用してください** 👉 0.7秒/画像

❌ CPU ローカル実行: 40-80秒/画像（非推奨）

#### 手動セットアップ（venv + uv）

```bash
# 1. リポジトリクローン
git clone https://github.com/Shion1124/anime-character-generator.git
cd anime-character-generator

# 2. 仮想環境作成
python3 -m venv venv
source venv/bin/activate

# 3. PyTorch をインストール（uv推奨、二進互換性保証）
if command -v uv &> /dev/null; then
    uv pip install torch torchvision torchaudio
else
    pip install torch torchvision torchaudio
fi

# 4. 依存関係インストール
pip install -r requirements.txt
pip install diffusers[torch] peft safetensors
```

#### テスト実行

```bash
# v2.0B本番版（推奨）
python character_generator_v2b.py --emotion happy --lcm

# v2.0開発版
python character_generator.py --emotion happy --lcm
```

**詳細なセットアップガイド**: 
- [LOCAL_SETUP_GUIDE.md](./LOCAL_SETUP_GUIDE.md) - ⚠️ CUDA GPU 環境向け
- [ENVIRONMENT_SETUP.md](./ENVIRONMENT_SETUP.md) - 包括的なガイド

### オプション C: Docker（完全な環境隔離）

```bash
docker run --rm -it \
    -v $(pwd):/workspace \
    pytorch/pytorch:2.1.0-cuda12.1-devel-ubuntu22.04 \
    bash -c "cd /workspace && pip install -r requirements.txt && python character_generator_v2b.py --emotion happy --lcm"
```

## 📚 コマンドラインオプション

### v2.0B (本番版)

```
python character_generator_v2b.py [OPTIONS]

オプション:
  --emotion {happy,angry,sad,surprised}  感情を指定
  --style STYLE                           スタイルを指定
  --all                                   すべてのバリエーション生成
  --device {cuda,mps,cpu}                 実行デバイス (デフォルト: auto)
  --lcm                                   LCM Scheduler で高速推論 ⚡
  --lora                                  LoRA (anime-character-lora_v1.5) を統合

使用例:
  # LCM高速推論（推奨）
  python character_generator_v2b.py --emotion happy --lcm
  
  # LoRA + LCM最適化
  python character_generator_v2b.py --all --lcm --lora
```

### v2.0 (開発版)

```
python character_generator.py [OPTIONS]

オプション:
  --emotion {happy,angry,sad,surprised}  感情を指定
  --style STYLE                           スタイルを指定
  --all                                   すべてのバリエーション生成
  --device {cuda,mps,cpu}                 実行デバイス (デフォルト: auto)
  --lcm                                   LCM Scheduler で高速推論 ⚡
  --lora                                  LoRA (anime-character-lora_v1.5) を統合
  --use-robust-prompt                     RobustPromptGenerator を使用 (Phase 1)

使用例:
  # 基本推論 (20 steps)
  python character_generator.py --emotion happy --style casual
  
  # LCM高速推論 (4 steps)
  python character_generator.py --emotion happy --lcm --lora
```**パフォーマンス**:
- **推論時間**: 0.7秒/画像（4ステップ LCM）
- **品質**: v1.5 と同等
- **環境**: macOS (MPS), Linux (CUDA), Windows (CUDA)

📖 **詳細**: [LOCAL_SETUP_GUIDE.md](./LOCAL_SETUP_GUIDE.md) を参照してください。

---

## 🔐 HuggingFace トークンについて

### v2.0B（推論版）- 読み取りトークン

[Google Colab\u1f4d3\uff1a](./anime_generator_colab_lora_v2b.ipynb) で推論実行時に **LoRA モデルをダウンロード** する場合：

```
トークンタイプ: 「Read」（読み取り）
取得URL: https://huggingface.co/settings/tokens
```

- LoRA モデルの読み込みに必要
- **読み取りのみのトークンで十分**
- 書き込みトークンでも動作

### v1.5（学習版）- 書き込みトークン

[Google Colab\u1f4d3\uff1a](./anime_generator_colab_lora_v1.5.ipynb) で学習後に **LoRA モデルを HuggingFace Hub にアップロード** する場合：

```
トークンタイプ: 「Write」（書き込み）← 重要！
取得URL: https://huggingface.co/settings/tokens
```

- LoRA モデルのアップロードに **必須**
- 読み取り専用トークンではアップロード失敗
- 「Write」トークンの作成時に「Write」を明確に選択してください

### トークン作成手順

1. [HuggingFace Settings → Access Tokens](https://huggingface.co/settings/tokens) にアクセス
2. **「+ Create new token」** をクリック
3. Token Name を入力（例: `anime-lora-read` または `anime-lora-write`）
4. **Token type で目的に応じて選択**：
   - **読み取りのみ**（推論）→ 「Read」
   - **アップロード**（学習）→ **「Write」**
5. **「Create token」** をクリック
6. トークンをコピーしてメモ帳に保存

> ⚠️ **セキュリティ注意**: トークンは GitHub などに公開しないでください！

---

## 💡 使用例

```python
from diffusers import StableDiffusionPipeline
import torch

# モデルロード
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# プロンプト定義
base = "1girl, anime character, masterpiece, high quality"
prompt = f"{base}, happy smile, cheerful, joyful"

# 生成実行
image = pipe(
    prompt=prompt,
    negative_prompt="low quality, blurry",
    num_inference_steps=20,
    guidance_scale=7.0,
    height=512,
    width=512
).images[0]

image.save("anime_character.png")
```

## 📁 プロジェクト構造

```
anime-character-generator/
├── README.md                          # このファイル
├── Improvement_Plan.md                # 今後の改善計画
├── requirements.txt                   # Python依存関係
├── LOCAL_SETUP_GUIDE.md               # ✅ ローカル推論セットアップガイド
├── anime_generator_colab_simple.ipynb # 推奨実行ノートブック (v1.0)
├── anime_generator_colab_lora_v1.5.ipynb # LoRA実装版 (v1.5)
├── anime_generator_colab_lora_v2b.ipynb # LCM蒸留版 ✅ (v2.0B/Colab)
├── character_generator.py             # プロダクション版スクリプト
├── character_generator_v2b.py         # ✅ 高速推論版 (v2.0B/ローカル)
├── outputs/
│   ├── emotion_results.png            # 感情グリッド合成 (v1.0)
│   ├── style_results.png              # スタイルグリッド合成 (v1.0)
│   ├── emotions/                      # 個別感情バリエーション画像
│   │   ├── character_happy.png
│   │   ├── character_angry.png
│   │   ├── character_sad.png
│   │   └── character_surprised.png
│   ├── styles/                        # 個別スタイルバリエーション画像（16パターン）
│   │   ├── character_with_hat.png
│   │   ├── character_with_earrings.png
│   │   ├── character_with_makeup.png
│   │   ├── ...
│   │   └── character_masterpiece.png
│   └── png/                           # ✅ バージョン別生成結果
│       ├── SDv1.5/                    # v1.0基本実装版
│       │   ├── emotion_results.png    # 感情グリッド
│       │   └── style_results.png      # スタイルグリッド
│       ├── SDv1.5+lora/               # v1.5 LoRA版
│       │   └── test_output.png        # テスト画像
│       └── SDv1.5+lora+LCM/           # ✅ v2.0B LCM蒸留版
│           ├── lcm_test_output_1.png  # テスト画像1 (0.99秒)
│           └── lcm_test_output_2.png  # テスト画像2 (0.83秒)
└── .gitignore
```

## 📊 生成結果サンプル

### 感情バリエーション（4パターン）

| Happy | Angry | Sad | Surprised |
|-------|-------|-----|-----------|
| ![Happy](outputs/emotions/character_happy.png) | ![Angry](outputs/emotions/character_angry.png) | ![Sad](outputs/emotions/character_sad.png) | ![Surprised](outputs/emotions/character_surprised.png) |

### スタイルバリエーション（6パターン）

| With Hat | With Earrings | Formal | Casual | With Makeup | Glasses |
|----------|---------------|--------|--------|-------------|---------|
| ![Hat](outputs/styles/character_with_hat.png) | ![Earrings](outputs/styles/character_with_earrings.png) | ![Formal](outputs/styles/character_formal.png) | ![Casual](outputs/styles/character_casual.png) | ![Makeup](outputs/styles/character_with_makeup.png) | ![Glasses](outputs/styles/character_glasses.png) |

## 🔧 カスタマイズ

### プロンプト修正

`anime_generator_colab_simple.ipynb` Step 5-6 や `character_generator.py` の辞書を編集：

```python
emotions = {
    "happy": "happy smile, cheerful, joyful",
    "angry": "angry expression, intense eyes",
    # さらに追加...
}

styles = {
    "with_hat": "wearing hat, stylish, fashionable",
    "formal": "wearing formal dress, elegant, professional",
    # 16パターンから自分好みに編集可能...
}
```

### グリッド合成のカスタマイズ

```python
# character_generator.py の generate_all() で調整可能

# 感情グリッド：2行2列（デフォルト）
self._create_grid_composite(emotion_images, "emotion_results", rows=2, cols=2)

# スタイルグリッド：2行4列（デフォルト）
self._create_grid_composite(style_images, "style_results", rows=2, cols=4)

# カスタマイズ例：4行4列グリッド
self._create_grid_composite(images, "custom_results", rows=4, cols=4, gap=15)
```

### 生成パラメータ調整

```python
image = pipe(
    prompt=prompt,
    negative_prompt="low quality, blurry",
    num_inference_steps=30,      # ↑ 品質向上（時間増加）
    guidance_scale=9.0,          # ↑ プロンプト厳格度
    height=768,                  # ↑ 解像度（メモリ増加）
    width=768
).images[0]
```

## 🎯 パフォーマンス

| 環境 | 1画像生成時間 | 実運用性 |
|------|-------------|---------|
| **Google Colab (T4 GPU)** | 3-5秒 | ⭐⭐⭐⭐⭐ 実用的 |
| Mac Mini (MPS) | 30-45秒 | ⭐⭐ 遅い |
| Mac Mini (CPU) | 2-3分 | ⭐ 非実用的 |

---

## 🧠 Phase 2: LoRA ファインチューニング

Stable Diffusion v1.5 を特定のスタイル（アニメ・印象派風）に特化させるため、**LoRA (Low-Rank Adaptation)** を使用したファインチューニングを実装しました。

### 📊 ステップ1: データセット収集

**Danbooru から 298 枚の画像を自動収集**

```bash
python scripts/download_danbooru.py --limit 60 --output training_data
```

**収集結果:**
- impressionist_style: 60 images (115 MB)
- oil_painting_aesthetic: 59 images (214 MB)
- sketch_aesthetic: 60 images (45 MB)
- soft_focus_landscape: 59 images (170 MB)
- pastel_softness: 60 images (115 MB)

**データセット検証:**
```bash
python scripts/validate_dataset.py --data-dir training_data
```

### 🎓 ステップ2: LoRA トレーニング

**Google Colab で実行（推奨）:**

```bash
# 依存パッケージインストール
!pip install -q diffusers transformers accelerate peft pillow torch tqdm safetensors

# train_lora.py をアップロード
# training_data/ ディレクトリをアップロード

# トレーニング実行（約1-2時間）
!python train_lora.py \
    --data_dir training_data \
    --output_dir lora_weights \
    --epochs 50 \
    --batch_size 1 \
    --learning_rate 1e-4
```

**出力:**
- `lora_weights/anime-impressionist-lora.safetensors` (~4 MB)
- `lora_weights/training_log.json` (学習履歴)

**ハイパーパラメータ:**
```
Model: Stable Diffusion v1.5
Learning Rate: 1e-4
Batch Size: 1 (T4 GPU制約)
Epochs: 50-100
LoRA Rank: 8
LoRA Alpha: 32
```

### 💾 ステップ3: 推論時に LoRA 適用

```python
from character_generator import AnimeCharacterGenerator

generator = AnimeCharacterGenerator()

# LoRA 重みを適用して生成
image = generator.generate_image(
    prompt="1girl, watercolor style, masterpiece",
    use_lora=True  # LoRA を有効化
)
image.save("output.png")
```

---

## 🎯 パフォーマンス

## � ファイル構成と対応関係

| バージョン | Python Script | Colab Notebook | ブログ対応 | 状態 |
|-----------|--------------|---|----------|------|
| **v1.0** | `character_generator_v1.py` | `anime_generator_colab_simple_v1.0.ipynb` | Day3-4前半完全対応 | ✅ 完成 |
| **v1.5** | `character_generator_v1_lora.py` | `anime_generator_colab_lora_v1.5.ipynb` | Day3-4後半（LoRA）対応 | ⚠️ 課題あり |
| **v2.0** | `character_generator.py` | `anime_generator_colab_lora_v2.0.ipynb` (準備中) | Phase 1-4 | 🔄 開発中 |

---

## 📚 ドキュメント体系

| ファイル | 説明 |
|---------|------|
| [README.md](./README.md) ← **このファイル** | プロジェクト概要・バージョン対応 |
| [Amendment.md](./Amendment.md) | v1.0/v1.5/v2.0 整合性修正計画 |
| [IMPLEMENTATION_ROADMAP.md](./IMPLEMENTATION_ROADMAP.md) | v2.0 Phase 1-4 実装計画 |
| [Improvement_Plan.md](./Improvement_Plan.md) | 理論設計・論文基盤 |
| [PHASE_1_PROMPT_OPTIMIZATION.md](./PHASE_1_PROMPT_OPTIMIZATION.md) | Phase 1 詳細設計（Gemini LLM） |
| [PHASE_2B_LCM_DISTILLATION.md](./PHASE_2B_LCM_DISTILLATION.md) | Phase 2B 詳細設計（LCM蒸留） |
| [PHASE_3_MULTIMODAL.md](./PHASE_3_MULTIMODAL.md) | Phase 3 詳細設計（マルチモーダル） |
| [PHASE_4_DEPLOYMENT.md](./PHASE_4_DEPLOYMENT.md) | Phase 4 詳細設計（デプロイ） |

---

## �📚 参考資料

- [Hugging Face Diffusers ドキュメント](https://huggingface.co/docs/diffusers)
- [Stable Diffusion モデルカード](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- [PyTorch 公式ガイド](https://pytorch.org/docs/stable/index.html)

## 🚀 次のステップ

詳細な改善計画と今後のロードマップは [Improvement_Plan.md](./Improvement_Plan.md) を参照してください。

- LLM統合によるプロンプトエンジニアリング
- LoRA ファインチューニング
- マルチモーダル推論
- 本番環境デプロイ

## 👤 Author

**Shion Shinzaki**
- GitHub: [@Shion1124](https://github.com/Shion1124)
- Blog: [shion.blog](https://shion.blog/)
- Email: soundpoem2022@gmail.com

## 📄 License

MIT License - see LICENSE.txt for details

---

**2026年2月17日** - v1.0 初版公開
