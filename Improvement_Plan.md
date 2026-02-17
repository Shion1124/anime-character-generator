# 🚀 Improvement_Plan.md

## プロジェクト進化ロードマップ

このドキュメントは、`anime-character-generator`の今後の改善計画をまとめています。

インタビューやポートフォリオの「次のステップ」として、以下の技術的拡張を検討中です。

---

## Phase 1: LLM統合によるプロンプト最適化

### 現状の課題

**v1.0（現在）**：プロンプト手動構築
```python
prompt = f"{base}, {emotion_desc}"
# 結果：プロンプト品質に依存。一般的な記述
```

**限界**：
- 感情表現が固定的
- キャラクター一貫性がない
- 複雑な指示対応が弱い

### Phase 1 解決案：Claude/GPT-4 API 活用

```python
import anthropic

client = anthropic.Anthropic()

def generate_optimized_prompt(emotion: str, style: str) -> str:
    """LLMで最適化されたプロンプト自動生成"""
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=150,
        messages=[{
            "role": "user",
            "content": f"""
            Generate a detailed anime character prompt for Stable Diffusion.
            Constraints:
            - Emotion: {emotion}
            - Style: {style}
            - Quality: high, detailed, masterpiece
            - Format: single line, comma-separated tags
            
            Output only the prompt, no explanation.
            """
        }]
    )
    return message.content[0].text

# 使用例
optimized_prompt = generate_optimized_prompt("happy", "formal dress")
# → "1girl, anime character, happy expression, enthusiastic, wearing formal elegant dress, 
#     high quality, detailed eyes, soft lighting, masterpiece, 8k"
```

### 期待される改善

| 指標 | v1.0 | Phase 1 |
|------|------|---------|
| プロンプト多様性 | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| キャラ一貫性 | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| 生成品質 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| API コスト | ¥0 | 約 ¥0.1-0.2/画像 |

### 実装ロードマップ

- [ ] Anthropic SDK 導入
- [ ] プロンプト生成ロジック実装
- [ ] キャッシング戦略（同じ組み合わせは再利用）
- [ ] A/B テスト（LLMなし vs あり）
- [ ] ブログ記事：「LLM + Diffusers の最適な組み合わせ」

---

## Phase 2: LoRA ファインチューニング

### 目的

汎用的な Stable Diffusion v1.5 ではなく、**独自のアニメスタイルに特化したモデル**を構築。

### 実装戦略

**ステップ 1: 学習データセット準備**
```
training_data/
├── happy_anime/
│   ├── img1.png
│   ├── img2.png
│   └── ...
├── angry_anime/
└── ...
合計 200-500 枚の高品質アニメ画像
```

**ステップ 2: Diffusers での LoRA 学習**

```python
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model

# LoRA設定
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["to_k", "to_v"],
    lora_dropout=0.1
)

# パイプライン準備
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.unet = get_peft_model(pipe.unet, lora_config)

# 学習ループ（数時間で完了）
# ...
```

### 成果物

- **カスタムLoRA重み** (~4MB) → HuggingFace Hub にアップロード
- **推論スクリプト** - LoRA統合版

```python
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.load_lora_weights("Shion1124/anime-character-lora")
```

---

## Phase 3: マルチモーダル推論

### 拡張機能案

**A) Image-to-Image 活用**

既存キャラクター画像から、異なる感情・スタイルへの自動変換。

```python
from diffusers import StableDiffusionImg2ImgPipeline

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(...)

# 現在の画像から派生バージョンを生成
derived = pipe(
    prompt="same character, angry expression",
    image=source_image,
    strength=0.7  # 0.0=元の画像, 1.0=フル再生成
).images[0]
```

**B) Controlnet 統合**

ユーザー指定のスケッチ・レイアウトから、それに従うキャラクター生成。

```python
from diffusers import StableDiffusionControlNetPipeline

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "lllyasviel/control_v11p_sd15_canny"
)
# スケッチ条件のもとで高品質生成
```

---

## Phase 4: 本番環境デプロイ

### クラウド展開案

**AWS Lambda + API Gateway**

```python
import json
from uuid import uuid4

def lambda_handler(event, context):
    emotion = event.get("emotion", "happy")
    style = event.get("style", "casual")
    
    # 生成処理
    image = generate_character(emotion, style)
    s3_key = f"output/{uuid4()}.png"
    
    # S3 に保存
    s3_client.put_object(Bucket="anime-gen", Key=s3_key, Body=image_bytes)
    
    return {
        "statusCode": 200,
        "body": json.dumps({
            "image_url": f"https://s3.amazonaws.com/anime-gen/{s3_key}"
        })
    }
```

### Web UI 構想

```html
<div id="generator">
  <select id="emotion">
    <option>Happy</option>
    <option>Angry</option>
    <option>Sad</option>
  </select>
  <select id="style">
    <option>Casual</option>
    <option>Formal</option>
  </select>
  <button onclick="generateCharacter()">Generate</button>
  <img id="result" />
</div>

<script>
async function generateCharacter() {
    const response = await fetch('/api/generate', {
        method: 'POST',
        body: JSON.stringify({
            emotion: document.getElementById('emotion').value,
            style: document.getElementById('style').value
        })
    });
    const data = await response.json();
    document.getElementById('result').src = data.image_url;
}
</script>
```

---

## スピリト企業向けアピール ポイント

### 技術的深さの表現

| Phase | スキル領域 | 実務経験 |
|-------|---------|--------|
| **v1.0** | Diffusers / PyTorch | ✅ 推論パイプライン |
| **Phase 1** | LLMプロンプトエンジニアリング | ✅ マルチモーダルAI |
| **Phase 2** | ファインチューニング / LoRA | ✅ カスタムモデル開発 |
| **Phase 3** | ControlNet / 条件付き生成 | ✅ 高度な制御技術 |
| **Phase 4** | MLOps / 本番環境構築 | ✅ 実運用経験 |

### 面接での活用

**質問**: "今後の技術的な方向性は？"

**回答テンプレート**：
```
現在はシンプルな推論実装ですが、スピリト様の R&D チームなら
以下のステップが自然だと考えます：

1. LLM統合で高度なプロンプト最適化
2. 社内アニメスタイルデータでLoRA学習
3. ControlNetで アニメータの「スケッチ → 完成画」への応用
4. 本番環境でのスケーラブルなAPI化

これは「プロトタイプから本番まで」の実務的なキャリアパスであり、
貴社の生成AI×アニメ制作 という領域にも直結します。
```

---

## 技術スタック拡張計画

```
現在：
PyTorch → Diffusers → Stable Diffusion v1.5

Phase 1-2 追加：
├── LLM: Claude API / GPT-4
├── LoRA: PEFT library
└── Storage: HuggingFace Hub

Phase 3 追加：
├── ControlNet: lllyasviel/ControlNet
├── Image Processing: OpenCV
└── Vision Transformers: CLIP Fine-tuning

Phase 4 追加：
├── 本番環境: AWS Lambda / GCP Cloud Run
├── API: FastAPI
├── Database: DynamoDB / PostgreSQL
└── Frontend: React + Next.js
```

---

## 優先順位と所要時間

| Phase | 優先度 | 所要時間 | ROI |
|-------|--------|--------|-----|
| v1.0（完了） | 🔴 必須 | 2日 | ⭐⭐⭐⭐⭐ |
| Phase 1 | 🟡 高 | 3-4日 | ⭐⭐⭐⭐ |
| Phase 2 | 🟡 高 | 5-7日 | ⭐⭐⭐ |
| Phase 3 | 🟢 中 | 4-5日 | ⭐⭐⭐ |
| Phase 4 | 🟢 中 | 2週間 | ⭐⭐ |

---

## 関連ブログ記事計画

1. **Day 1-2**: Stable Diffusion 基礎（完成）
2. **Day 3**: PyTorch + Diffusers 実装解説 + 生成結果
3. **Day 4**: GitHub公開記事
4. **Future Blog**:
   - 🔄 「LLMでプロンプトを最適化する」
   - 🔄 「Stable Diffusion をファインチューニングする」
   - 🔄 「ControlNet で スケッチから画像生成」

---

## 参考リソース

- [Hugging Face PEFT (LoRA)](https://github.com/huggingface/peft)
- [ControlNet Official](https://github.com/lllyasviel/ControlNet)
- [Anthropic Claude API](https://docs.anthropic.com/)
- [AWS Lambda MLOps](https://aws.amazon.com/jp/blogs/machine-learning/)

---

**最終更新**: 2026年2月17日

**次のレビュー**: Phase 1 実装開始時
