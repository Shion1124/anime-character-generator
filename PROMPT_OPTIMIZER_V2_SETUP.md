# prompt_optimizer_v2.py セットアップガイド

## 概要

`prompt_optimizer_v2.py` v2 は、**Google Generative AI (Gemini)** または **HuggingFace ローカルモデル** に対応した LLM ベースのプロンプト生成エンジンです。

### 対応バックエンド

| バックエンド | 特徴 | 推奨用途 |
|------------|------|--------|
| **Google Generative AI** ☁️ | クラウド型、高速、Gemini採用 | 開発・試験、速度重視 |
| **HuggingFace Local** 🖥️ | オンプレミス、Qwen/Swallow等 | オフライン、プライバシー重視 |

---

## 📋 セットアップ手順

### 1️⃣ 環境変数確認（`.env`ファイル）

```bash
# プロジェクトルートで確認
cat .env
```

期待される内容：
```ini
# Google API キー
Google_Api_Key=AIzaSyBJNm_8h2nB1-Do_FTms1EyV_x3wUp18R0

# HuggingFace モデル（ローカル実行時）
HUGGINGFACE_MODEL=tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2
HF_CACHE_DIR=~/.cache/huggingface/hub
```

---

### 2️⃣ Option A: Google Generative API を使用

#### インストール

```bash
pip install google-generativeai
```

#### テスト実行

```python
from prompt_optimizer_v2 import RobustPromptGenerator

# Google API を使用（デフォルト）
generator = RobustPromptGenerator(use_google_api=True)

result = generator.generate_prompt(
    "happy anime girl with pink hair",
    use_lcm=True
)

print(f"✨ Prompt: {result['positive_prompt']}")
print(f"⚙️  LCM Settings: {result['lcm_settings']}")
print(f"📊 Confidence: {result['confidence']:.2f}")
```

**利点**：
- ✅ 即座に利用可能（API キーのみ必要）
- ✅ 高品質プロンプト（Gemini 採用）
- ✅ 低レイテンシ

**制限**：
- ❌ API コスト発生（トークンベース）
- ❌ インターネット接続必須

---

### 3️⃣ Option B: HuggingFace ローカルモデルを使用

#### インストール

```bash
pip install transformers torch
```

#### テスト実行

```python
from prompt_optimizer_v2 import RobustPromptGenerator

# HuggingFace ローカルモデルを使用
generator = RobustPromptGenerator(use_google_api=False)

result = generator.generate_prompt(
    "calm anime girl in a peaceful garden",
    use_lcm=True,
    controlnet_mode="lineart"
)

print(f"✨ Prompt: {result['positive_prompt']}")
print(f"🎮 ControlNet: {result['controlnet_settings']}")
```

**利点**：
- ✅ API コスト 0 円
- ✅ インターネット不要（オフライン利用可）
- ✅ カスタマイズ可能（独自モデル選択可）

**制限**：
- ❌ VRAM 必要（8GB+ 推奨）
- ❌ 初回実行時 モデル ダウンロード時間長い

**推奨モデル**：
```
tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2  # 8B, 日本語対応
```

---

## 🚀 実行例

### 例 1: LCM-LoRA 対応プロンプト生成（Google API）

```bash
python3 << 'EOF'
from prompt_optimizer_v2 import RobustPromptGenerator

generator = RobustPromptGenerator(use_google_api=True)

# LCM-LoRA 用に最適化されたプロンプト
result = generator.generate_prompt(
    description="happy anime girl with blue eyes",
    use_lcm=True,
    controlnet_mode=None,
    quality_level="masterpiece"
)

print("=" * 60)
print("✨ Positive Prompt (LCM-LoRA 用)")
print("=" * 60)
print(result['positive_prompt'])
print("\n" + "=" * 60)
print("❌ Negative Prompt")
print("=" * 60)
print(result['negative_prompt'])
print("\n" + "=" * 60)
print("⚙️  LCM Settings")
print("=" * 60)
print(f"  - Guidance Scale: {result['lcm_settings']['guidance_scale']}")
print(f"  - Inference Steps: {result['lcm_settings']['num_inference_steps']}")
print(f"  - Scheduler: {result['lcm_settings']['scheduler']}")
print(f"\n📊 Confidence Score: {result['confidence']:.2f}")
EOF
```

**期待される出力**：
```
============================================================
✨ Positive Prompt (LCM-LoRA 用)
============================================================
1girl, happy expression, blue eyes, anime character, masterpiece, best quality, ...

============================================================
❌ Negative Prompt
============================================================
low quality, blurry, deformed, ...

============================================================
⚙️  LCM Settings
============================================================
  - Guidance Scale: 1.5
  - Inference Steps: 4
  - Scheduler: LCMScheduler

📊 Confidence Score: 0.87
```

---

### 例 2: ControlNet + LCM 統合（HuggingFace ローカル）

```bash
python3 << 'EOF'
from prompt_optimizer_v2 import RobustPromptGenerator

generator = RobustPromptGenerator(use_google_api=False)  # ローカルモデル

# ControlNet (Lineart) 対応プロンプト
result = generator.generate_prompt(
    description="elegant anime girl in flowing dress",
    use_lcm=True,
    controlnet_mode="lineart",
    controlnet_conditioning_scale=0.8
)

print("ControlNet (Lineart) + LCM 生成設定:")
print(f"  Conditioning Scale: {result['controlnet_settings']['conditioning_scale']}")
print(f"\n生成プロンプト: {result['positive_prompt']}")
EOF
```

---

### 例 3: バッチプロンプト生成

```bash
python3 << 'EOF'
from prompt_optimizer_v2 import RobustPromptGenerator

generator = RobustPromptGenerator(use_google_api=True)

# 複数の描写から一括生成
descriptions = [
    "happy anime girl",
    "sad anime boy",
    "angry warrior character",
    "calm peaceful person"
]

results = generator.batch_generate(
    descriptions=descriptions,
    use_lcm=True
)

print(f"✅ Generated {len(results)} prompts\n")
for i, result in enumerate(results, 1):
    print(f"{i}. {result['positive_prompt'][:60]}... (confidence: {result['confidence']:.2f})")
EOF
```

---

## 🔧 トラブルシューティング

### 問題 1: `Google_Api_Key not set in .env`

**原因**：`Google_Api_Key` が `.env` に設定されていない

**解決**：
```bash
# .env を編集
echo "" >> .env
echo "Google_Api_Key=YOUR_API_KEY_HERE" >> .env

# APIキーを.env に追加
source .env
```

---

### 問題 2: `google-generativeai not installed`

**原因**：Google API ライブラリがインストールされていない

**解決**：
```bash
pip install google-generativeai
```

---

### 問題 3: HuggingFace モデルが遅い

**原因**：VRAM不足、またはCPU実行

**確認**：
```bash
python3 << 'EOF'
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
EOF
```

**改善策**：
- Google API に切り替え（Option A）
- より軽量なモデル使用：
  ```python
  os.environ["HUGGINGFACE_MODEL"] = "gpt2"  # 軽量テスト用
  ```

---

### 問題 4: JSON パースエラー

**原因**：API が JSON 形式でない応答を返した

**対応**：自動フォールバック機能が発動
- フォールバック プロンプトが返される（confidence: 0.5）
- ログに `⚠️  JSON パースエラー。フォールバック使用` と表示

---

## 📊 パフォーマンス比較

| 指標 | Google API | HuggingFace Local |
|-----|-----------|------------------|
| 初回実行時間 | ~2秒 | ~30秒（モデルDL） |
| 通常実行時間 | ~1-2秒 | ~5-10秒 |
| API コスト | $0.00075/プロンプト | ¥0 |
| インターネット | 必須 | 不要 |
| VRAM | 不要 | 8GB+ |
| 品質（感情タグ） | 優秀 | 良好 |

---

## 🔌 Colab ノートブック統合例

[anime_generator_colab_lora_v2b.ipynb](./anime_generator_colab_lora_v2b.ipynb) の Step 1.5 に組み込む場合：

```python
# === Step 1.5: プロンプト最適化 ===

# Google API キーが設定されているか確認
import os
from pathlib import Path

api_key = os.getenv("Google_Api_Key")
if not api_key:
    print("⚠️  Google_Api_Key not set. Set via secrets or .env")

# prompt_optimizer をインストール・実行
from prompt_optimizer_v2 import RobustPromptGenerator

generator = RobustPromptGenerator(use_google_api=True)

# テスト実行
test_result = generator.generate_prompt(
    "anime girl, happy expression",
    use_lcm=True
)

print(f"✨ Prompt: {test_result['positive_prompt']}")
print(f"📊 Confidence: {test_result['confidence']:.2f}")
```

---

## 📚 関連ドキュメント

- [LCM-LoRA 統合ガイド](./PHASE_3_CONTROLNET_AND_PROMPT_INTEGRATION.md)
- [実装ロードマップ](./EXECUTION_PLAN_V2B_PHASE3.md)
- [理論検証レポート](./THEORY_VERIFICATION_REPORT.md)

---

## 🎯 次のステップ

✅ **このステップ完了後**：
1. anime_generator_colab_lora_v2b.ipynb に統合
2. ControlNet 統合テスト（Phase 3）
3. HuggingFace Collections リリース準備

