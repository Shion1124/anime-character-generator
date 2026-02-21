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

### ⚠️ v1.5: LoRA ファインチューニング版（試行版・課題あり）

| 特性 | 詳細 |
|------|------|
| **ファイル** | `character_generator_v1_lora.py`<br/>`anime_generator_colab_lora_v1.5.ipynb` |
| **説明** | ブログのLoRA実装セクション準拠<br/>試行錯誤版として保持 |
| **機能** | v1.0 + LoRA ファインチューニング<br/>アニメスタイルへの特化 |
| **速度** | 3.8秒/画像 (v1.0と同じ) |
| **既知の課題** | ⚠️ 以下4つの課題あり（v2.0で解決予定） |

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

### ✅ v2.0: 学術的改善版（Phase 1-4 実装中）

| 特性 | 詳細 |
|------|------|
| **ファイル** | `character_generator.py`<br/>`anime_generator_colab_lora_v2.0.ipynb` (準備中) |
| **説明** | Phase 1-4 による段階的改善<br/>論文ベース実装 |
| **改善内容** | 本ドキュメント下部を参照 |
| **状態** | 🔄 実装フェーズ |

**Phase 計画**:

| Phase | 目的 | 改善項目 | 期限 |
|-------|------|---------|------|
| **Phase 1** | Gao et al. 脆弱性対応 | Gemini LLM 多層冗長プロンプト | Week 1-2 |
| **Phase 2A** | メモリ効率化 | 改善されたLoRA実装 | Week 2-3 |
| **Phase 2B** | 推論高速化 | LCM 蒸留（12x高速化） | Week 3-4 |
| **Phase 3** | マルチモーダル対応 | Image-to-Image + ControlNet | Week 4-5 |
| **Phase 4** | 本番環境対応 | Streamlit UI + FastAPI + Docker | Week 5-6 |

詳細: [IMPLEMENTATION_ROADMAP.md](./IMPLEMENTATION_ROADMAP.md)

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

### オプション B: ローカル実行（Mac/Linux）

```bash
# 1. リポジトリクローン
git clone https://github.com/Shion1124/anime-character-generator.git
cd anime-character-generator

# 2. 仮想環境作成
python3.11 -m venv venv
source venv/bin/activate

# 3. 依存関係インストール
pip install -r requirements.txt

# 4. 実行
python character_generator.py
```

**注意**：ローカルGPU（NVIDIA CUDAまたはApple Silicon MPS）が必要です。CPU-onlyの場合、生成時間が著しく増加します。

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
├── anime_generator_colab_simple.ipynb # 推奨実行ノートブック
├── anime_generator_colab.ipynb        # 詳細版ノートブック
├── character_generator.py             # プロダクション版スクリプト
├── outputs/
│   ├── emotion_results_v1.png         # 感情グリッド合成（2x2）
│   ├── emotion_results_v2.png         # 自動バージョン管理
│   ├── style_results_v1.png           # スタイルグリッド合成（2x4）
│   ├── style_results_v2.png           # 自動バージョン管理
│   ├── emotions/                      # 個別感情バリエーション画像
│   │   ├── character_happy.png
│   │   ├── character_angry.png
│   │   ├── character_sad.png
│   │   └── character_surprised.png
│   └── styles/                        # 個別スタイルバリエーション画像（16パターン）
│       ├── character_with_hat.png
│       ├── character_with_earrings.png
│       ├── character_with_makeup.png
│       ├── ...
│       └── character_masterpiece.png
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
