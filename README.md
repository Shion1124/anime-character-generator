# 🎨 anime-character-generator

Stable Diffusion + PyTorch を活用した、**アニメキャラクター自動生成システム**。複数の感情・スタイルバリエーションを一度に生成できます。

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

## 📚 参考資料

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
