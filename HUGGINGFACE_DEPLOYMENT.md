# HuggingFace Hub へのモデルアップロードガイド

このドキュメントは、LoRA モデルを HuggingFace Hub にアップロードするプロセスを説明します。

## 📋 前提条件

### 1. HuggingFace アカウント作成

- [HuggingFace](https://huggingface.co/) でアカウントを作成
- **無料**でモデルをアップロード・共有できます

### 2. API トークン生成

1. [Settings → Access Tokens](https://huggingface.co/settings/tokens) にアクセス
2. **New Token** をクリック
3. トークン名を入力（例：`anime-lora-upload`）
4. **Role** を `write` に設定
5. トークンをコピーして安全に保存

### 3. 必要なパッケージインストール

```bash
pip install huggingface-hub diffusers peft
```

## 🚀 アップロード方法

### 方法 1: Python スクリプト（推奨）

```bash
# 環境変数でトークンを設定
export HF_TOKEN="hf_xxxxxxxxxxxxx"

# アップロード実行
python upload_to_huggingface.py \
    --model-path ./anime-lora-weights \
    --repo-name anime-character-lora
```

### 方法 2: コマンドラインオプションで指定

```bash
python upload_to_huggingface.py \
    --model-path ./anime-lora-weights \
    --repo-name anime-character-lora \
    --hf-token "hf_xxxxxxxxxxxxx"
```

### 方法 3: HuggingFace CLI ログイン

```bash
# 初回のみ実行（対話的にログイン）
huggingface-cli login

# その後は以下で実行可能
python upload_to_huggingface.py \
    --model-path ./anime-lora-weights \
    --repo-name anime-character-lora
```

### 方法 4: Bash スクリプト使用

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxx"
bash scripts/upload_to_huggingface.sh ./anime-lora-weights anime-character-lora
```

## 📝 よくあるオプション

### プライベートリポジトリとして公開

```bash
python upload_to_huggingface.py \
    --model-path ./anime-lora-weights \
    --repo-name anime-character-lora \
    --private
```

### オーガニゼーション配下にアップロード

```bash
python upload_to_huggingface.py \
    --model-path ./anime-lora-weights \
    --repo-name anime-character-lora \
    --org-name my-organization
```

## ✅ 確認

アップロード後、以下で確認できます：

```
https://huggingface.co/YOUR_USERNAME/anime-character-lora
```

### モデルの使用

アップロード後、以下のコードでモデルを使用できます：

```python
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel

# ベースモデルをロード
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)

# HuggingFace Hub から LoRA をロード
pipe.unet = PeftModel.from_pretrained(
    pipe.unet,
    "YOUR_USERNAME/anime-character-lora",  # リポジトリID
    adapter_name="anime_lora"
)

pipe = pipe.to("cuda")

# 画像生成
image = pipe(
    prompt="1girl, anime character, masterpiece, high quality",
    num_inference_steps=20,
    guidance_scale=7.5,
    height=512,
    width=512
).images[0]

image.save("output.png")
```

## 📊 アップロード内容

スクリプトは以下をアップロードします：

| ファイル | 説明 |
|---------|------|
| `adapter_config.json` | LoRA 設定（ランク、アルファ値など） |
| `adapter_model.bin` | LoRA 重み本体 |
| `README.md` | モデルカード（説明・使用方法） |

### ファイルサイズ例

```
adapter_config.json:  ~1 KB
adapter_model.bin:    ~2-3 MB
README.md:            ~30 KB
合計:                 ~3 MB
```

## 🐛 トラブルシューティング

### エラー: `HF_TOKEN が見つかりません`

**解決方法:**
```bash
export HF_TOKEN="hf_xxxxxxxxxxxxx"
```

または `--hf-token` オプションを使用

### エラー: `リポジトリ作成エラー`

**原因:**
- HuggingFace アカウントが有効でない
- リポジトリ名が既に別ユーザーで使用されている（拡張子や番号を追加）

**解決方法:**
```bash
python upload_to_huggingface.py \
    --model-path ./anime-lora-weights \
    --repo-name anime-character-lora-v2  # 別名を試す
```

### エラー: `認証エラー`

**解決方法:**
1. トークンが有効か確認
2. トークンをリジェネレート: https://huggingface.co/settings/tokens
3. 新しいトークンで再試行

### ネットワークエラー

**解決方法:**
- インターネット接続を確認
- ファイアウォール設定を確認
- 別の時間に再試行

## 📚 参考資料

- [HuggingFace Hub ドキュメント](https://huggingface.co/docs/hub/index)
- [huggingface_hub ライブラリ](https://github.com/huggingface/huggingface_hub)
- [モデルカード仕様](https://huggingface.co/docs/hub/model-cards)
- [PEFT ライブラリ](https://github.com/huggingface/peft)
- [Diffusers ライブラリ](https://huggingface.co/docs/diffusers/)

## 🎯 次のステップ

1. **モデルの共有**: Reddit、Twitter、GitHub Discussions で公開
2. **フィードバック収集**: Issue/Discussion で改善提案を集約
3. **改善版リリース**: より多くのスタイル・エフェクトで新版をリリース
4. **デリバティブ**: コミュニティの派生モデルをフォーク

## 📞 サポート

アップロード中に問題が発生した場合：

1. **スクリプトのヘルプを確認**
   ```bash
   python upload_to_huggingface.py --help
   ```

2. **GitHub Issues で報告**
   https://github.com/Shion1124/anime-character-generator/issues

3. **HuggingFace Community Forum**
   https://huggingface.co/spaces

---

**作成日:** 2026年2月18日  
**スクリプトバージョン:** v1.0
