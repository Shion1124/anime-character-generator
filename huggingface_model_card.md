# 🎨 anime-impressionist-lora

**文体:** 印象派風、水彩画風、アニメキャラクター  
**モデル:** Stable Diffusion v1.5 + LoRA (Low-Rank Adaptation)  
**ファイルサイズ:** 約 4 MB (軽量・高速)

---

## 📊 モデル情報

| 項目 | 詳細 |
|------|------|
| **ベースモデル** | `runwayml/stable-diffusion-v1-5` |
| **ファインチューニング手法** | LoRA (PEFT - Parameter-Efficient Fine-Tuning) |
| **トレーニングデータ** | Danbooru 300 枚 (5 スタイル) × 660 MB |
| **学習時間** | 約 1-2 時間 (Colab T4 GPU、50エポック) |
| **LoRA ランク** | 8 |
| **LoRA アルファ値** | 32 |
| **学習率** | 1e-4 |
| **バッチサイズ** | 1 (T4 GPU メモリ最適化) |
| **最適化手法** | AdamW |

---

## 🚀 使用方法

### 方法 1: diffusers + PEFT（推奨、最も簡単）

```python
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel

# ベースモデルをロード
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)

# LoRA 重みをロード（HuggingFace Hub または ローカルパス）
pipe.unet = PeftModel.from_pretrained(
    pipe.unet,
    "YOUR_USERNAME/anime-impressionist-lora",  # または "./lora_weights"
    adapter_name="anime_lora"
)

pipe = pipe.to("cuda")

# 画像生成
prompt = "1girl, watercolor painting style, masterpiece, high quality, anime"
negative_prompt = "low quality, nsfw, blurry"

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=20,
    guidance_scale=7.0,
    height=512,
    width=512
).images[0]

image.save("output.png")
```

### 方法 2: anime-character-generator CLI

```bash
# このプロジェクトの CLI で使用
python character_generator.py \
  --lora-path ./anime-impressionist-lora.safetensors \
  --use-lora \
  --emotion happy \
  --style anime
```

### 方法 3: HuggingFace Inference API

```python
from huggingface_hub import InferenceClient

client = InferenceClient()
image = client.text_to_image(
    prompt="1girl, watercolor painting, masterpiece",
    model="YOUR_USERNAME/anime-impressionist-lora"
)
```

---

## 🎨 生成例とプロンプト推奨設定

### プロンプトテンプレート

```
# 基本形（シンプル）
1girl, anime character, masterpiece, high quality

# スタイル指定（推奨される文体）
1girl, watercolor painting style, soft focus, impressionist, anime, masterpiece

# 詳細指定（より高い品質を求める場合）
1girl, beautiful detailed face, long brown hair, watercolor aesthetic, 
soft blush, warm lighting, bokeh background, masterpiece, best quality
```

### 推奨生成パラメータ

| パラメータ | 推奨値 | 説明 |
|-----------|--------|------|
| `num_inference_steps` | 20-30 | ステップ数（多いほど詳細で遅くなる） |
| `guidance_scale` | 7.0-8.5 | プロンプト従順性（高いほどプロンプトに従う） |
| `height` / `width` | 512×512 | 最適な出力サイズ |
| `negative_prompt` | "low quality, nsfw, blurry" | 除外ワード |
| `seed` | -1 (ランダム) | 再現性設定（同じseedで同じ結果） |

### 推奨ネガティブプロンプト

```
"low quality, worst quality, blurry, distorted, watermark, 
error, nsfw, extra limbs, missing limbs, ugly, bad anatomy"
```

---

## 📈 トレーニング詳細

### データセット構成

Danbooru より自動収集した高品質アニメ画像セット：

| スタイル | 枚数 | タグ | 特徴 |
|---------|------|------|------|
| **impressionist_style** | 60枚 | fantasy, impressionist | ファンタジー、印象派風 |
| **soft_focus_landscape** | 59枚 | landscape, soft focus | ランドスケープ、ソフトフォーカス |
| **oil_painting_aesthetic** | 59枚 | scenery, oil painting | 風景、油絵調 |
| **sketch_aesthetic** | 60枚 | sketch, line art | スケッチ、線画 |
| **pastel_softness** | 60枚 | pastel, soft colors | パステルカラー、柔らかい色合い |

**合計:** 298 枚、660 MB

### ハイパーパラメータ

```python
# モデル設定
model_id = "runwayml/stable-diffusion-v1-5"
lora_rank = 8
lora_alpha = 32
lora_dropout = 0.1
target_modules = ["to_k", "to_v", "to_q"]  # 注意層のキー・値・クエリ

# トレーニング設定
learning_rate = 1e-4
batch_size = 1
num_epochs = 50
gradient_accumulation_steps = 1
mixed_precision = "fp16"  # T4 GPU最適化

# 最適化設定
optimizer = "AdamW"
weight_decay = 0.01
max_grad_norm = 1.0
```

### 学習曲線

トレーニングプロセス：
- **初期損失:** ~0.3
- **最終損失:** ~0.05
- **改善率:** 約 83%
- **収束速度:** 10-15 エポック後に安定

---

## 🎯 モデルの特徴

### ✅ 得意な生成対象

- 🎨 **アニメキャラクター生成**: 顔、髪、表情の詳細表現
- 🖌️ **水彩・印象派風**: 柔らかい色合い、ぼかし効果
- 🌅 **風景・背景**: ランドスケープ、光源表現
- ✨ **装飾・エフェクト**: ボケ、グロー、光の筋
- 📐 **スケッチ・線画**: ペン画、スケッチ調

### ⚠️ 既知の限界

- **3D リアル画像**: Stable Diffusion v1.5 が理想的でないため精度低下
- **複雑な構図**: 3人以上のキャラクター、複雑な配置
- **テキスト生成**: 画像内のテキスト（プロンプトでも出力精度低い）
- **極端なスタイル**: 大幅に異なるアート様式への適応が限定的
- **高精度の手・指**: 元々のStable Diffusionの限界

---

## 📝 トレーニングコード（リファレンス）

詳細な実装は以下を参照：

```python
# train_lora.py より
class LoRATrainer:
    def __init__(self, model_id, lora_rank=8, lora_alpha=32):
        self.model = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        
        # LoRA 設定
        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["to_k", "to_v", "to_q"],
            lora_dropout=0.1
        )
        self.unet = get_peft_model(self.unet, peft_config)
```

完全なトレーニングスクリプトは [anime-character-generator](https://github.com/Shion1124/anime-character-generator) リポジトリでご確認ください。

---

## 📋 ライセンス

このモデルは以下に基づいています：

1. **Stable Diffusion v1.5**: [OpenRAIL License](https://huggingface.co/spaces/CompVis/stable-diffusion-license)
   - 研究、商用利用が可能
   - 責任ある利用が求められる

2. **LoRA ファインチューニング**: オリジナル実装
   - 同じくOpenRAIL準拠
   - Danbooru データセット利用（CC0ライセンス）

**使用条件:**
- ✅ 学術研究
- ✅ 創作支援、イラスト生成
- ✅ エンターテイメント
- ✅ 個人プロジェクト
- ⚠️ 商用利用（OpenRAILの規約に従うこと）
- ❌ 違法コンテンツ生成
- ❌ 個人を害する明確な意図での利用

---

## 🤝 フィードバック・改善

Issues や discussions での報告をお待ちしています：

- 生成品質の改善提案
- バグ報告（出力エラーなど）
- 使用例の共有
- 新しいスタイル提案

GitHub: [@Shion1124/anime-character-generator](https://github.com/Shion1124/anime-character-generator)

---

## 📚 参考資料

- **プロジェクト GitHub**: [anime-character-generator](https://github.com/Shion1124/anime-character-generator)
- **開発ガイド**: [dev_peft.md](https://github.com/Shion1124/anime-character-generator/blob/main/dev_peft.md)
- **PEFT Documentation**: [huggingface/peft](https://github.com/huggingface/peft)
- **Stable Diffusion v1.5**: [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- **Danbooru**: [danbooru.donmai.us](https://danbooru.donmai.us)

---

## 👤 作者

**Shion Shinzaki**
- GitHub: [@Shion1124](https://github.com/Shion1124)
- HuggingFace: [@Shion1124](https://huggingface.co/Shion1124)

**プロジェクト開始日:** 2026年2月18日  
**LoRA リリース日:** 2026年2月18日  
**最終更新:** 2026年2月18日

---

**🎉 このモデルの利用をお楽しみください！**

ご質問や提案は [GitHub Issues](https://github.com/Shion1124/anime-character-generator/issues) までお願いします。
