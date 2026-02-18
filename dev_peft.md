# 🎨 LoRA ファインチューニング開発手順書

**プロジェクト**: anime-character-generator Phase 2 実装  
**目標**: 印象派風朧げなアニメーション画像を学習した LoRA モデルの構築  
**推定期間**: 7-10日  
**最終成果物**: `anime-impressionist-lora.safetensors` (+ブログ記事)

---

## 📋 目的と戦略

### なぜ LoRA ファインチューニングが必要か？

| 現状（v1.0） | Phase 2（LoRA適用） |
|-----------|------------------|
| 汎用 Stable Diffusion v1.5 | **独自スタイルへの特化** |
| アニメ品質: ⭐⭐⭐ | アニメ品質: ⭐⭐⭐⭐⭐ |
| スタイル一貫性: ⭐⭐ | スタイル一貫性: ⭐⭐⭐⭐⭐ |
| 差別化なし | **独自ブランド化** |

### 学習対象スタイル

```
追求するビジュアルスタイル：
- 印象派画家のような朧げなタッチ
- 小説の風景描写のような詩的雰囲気
- 水彩画的・油彩画的な質感
- ソフトフォーカスエレガント美学
```

---

## 🗂️ Phase 2 実装パイプライン

```
Step 1: データセット準備 (Days 1-2)
   ├─ Danbooru から 300 枚収集
   ├─ metadata.json 作成
   └─ データセット検証
   
Step 2: LoRA 学習スクリプト (Days 3-4)
   ├─ train_lora.py 実装
   ├─ ハイパーパラメータ調整
   └─ 学習環境構築（Google Colab）
   
Step 3: モデル学習 (Days 5-6)
   ├─ 学習実行（T4 GPU, 1-2時間）
   ├─ 重み保存（~4MB）
   └─ 学習曲線分析
   
Step 4: 推論統合 (Days 7)
   ├─ character_generator.py へ LoRA ロード機能追加
   ├─ テスト画像生成
   └─ 品質評価
   
Step 5: ドキュメント化 (Days 8-9)
   ├─ README.md 更新
   ├─ ブログ記事執筆
   └─ GitHub push
   
Step 6: 本番化 (Day 10)
   ├─ HuggingFace Hub アップロード
   ├─ 公開
   └─ フィードバック収集
```

---

## 📥 Step 1: データセット準備

### 1.1 ディレクトリ構造

```bash
anime-character-generator/
├── training_data/                    # ← 新規作成
│   ├── impressionist_style/          # 印象派的（100-150枚）
│   ├── watercolor_aesthetic/         # 水彩画的（100-150枚）
│   ├── soft_focus_landscape/         # 朧げな風景（50-100枚）
│   ├── metadata.json                 # タグ情報
│   └── download_log.txt              # ダウンロードログ
├── lora_weights/                     # ← 新規作成
│   └── anime-impressionist-lora.safetensors  # 学習後ここに保存
├── dev_peft.md                       # このファイル
└── train_lora.py                     # 学習スクリプト（後で実装）
```

### 1.2 Danbooru からの画像収集

#### 方法 A: API スクリプト（推奨）

```bash
# 必要なパッケージ
pip install requests pillow tqdm
```

**実装ファイル: `scripts/download_danbooru.py`**

```python
#!/usr/bin/env python3
"""
Danbooru から印象派風アニメ画像を収集するスクリプト

使用例:
    python scripts/download_danbooru.py --output training_data --limit 300
"""

import requests
import json
import os
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import time

class DanbooruDownloader:
    """Danbooru 画像ダウンロード"""
    
    BASE_URL = "https://danbooru.donmai.us/posts.json"
    
    # スタイル別タグ定義
    STYLE_TAGS = {
        "impressionist_style": [
            "watercolor", "impressionist_style", "-lowres"
        ],
        "soft_focus_landscape": [
            "soft_focus", "landscape", "anime", "-fake_photorealism", "-lowres"
        ],
        "oil_painting_aesthetic": [
            "oil_painting_style", "aesthetic", "anime", "-lowres"
        ],
        "sketch_aesthetic": [
            "sketch", "anime_sketch", "aesthetic", "-lowres"
        ],
        "pastel_softness": [
            "pastel_colors", "soft_shading", "anime", "-lowres"
        ]
    }
    
    def __init__(self, output_dir: str = "training_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.metadata = []
        self.download_log = []
    
    def download_images(self, limit_per_style: int = 60):
        """スタイル別に画像をダウンロード"""
        
        total_downloaded = 0
        
        for style_name, tags in self.STYLE_TAGS.items():
            print(f"\n📥 Downloading: {style_name}")
            print(f"   Tags: {', '.join(tags)}")
            
            # スタイル別ディレクトリ作成
            style_dir = self.output_dir / style_name
            style_dir.mkdir(exist_ok=True)
            
            downloaded = self._download_style(style_name, tags, style_dir, limit_per_style)
            total_downloaded += downloaded
            
            print(f"   ✅ {downloaded}/{limit_per_style} downloaded")
            time.sleep(2)  # API リクエスト間隔
        
        print(f"\n✅ Total: {total_downloaded} images downloaded")
        self._save_metadata()
        return total_downloaded
    
    def _download_style(self, style_name: str, tags: List[str], 
                       output_dir: Path, limit: int) -> int:
        """特定スタイルの画像をダウンロード"""
        
        tag_string = " ".join(tags)
        downloaded = 0
        page = 1
        
        with tqdm(total=limit, desc=style_name) as pbar:
            while downloaded < limit:
                try:
                    response = requests.get(
                        self.BASE_URL,
                        params={
                            "tags": tag_string,
                            "limit": 200,
                            "page": page
                        },
                        timeout=10
                    )
                    response.raise_for_status()
                    
                    images = response.json()
                    if not images:
                        break
                    
                    for image in images:
                        if downloaded >= limit:
                            break
                        
                        # ファイル URL 取得
                        file_url = None
                        if "file_url" in image:
                            file_url = image["file_url"]
                        elif "large_file_url" in image:
                            file_url = image["large_file_url"]
                        
                        if not file_url:
                            continue
                        
                        # 画像ダウンロード
                        try:
                            img_response = requests.get(file_url, timeout=10)
                            img_response.raise_for_status()
                            
                            # ファイル保存
                            filename = f"{style_name}_{downloaded:03d}.png"
                            filepath = output_dir / filename
                            
                            with open(filepath, "wb") as f:
                                f.write(img_response.content)
                            
                            # メタデータ記録
                            self.metadata.append({
                                "file": str(filepath.relative_to(self.output_dir)),
                                "style": style_name,
                                "tags": image.get("tag_string_general", "").split(),
                                "width": image.get("image_width"),
                                "height": image.get("image_height")
                            })
                            
                            self.download_log.append(f"✅ Downloaded: {filename}")
                            downloaded += 1
                            pbar.update(1)
                        
                        except Exception as e:
                            self.download_log.append(f"❌ Failed: {file_url} - {e}")
                            continue
                    
                    page += 1
                
                except Exception as e:
                    print(f"   ⚠️  API Error: {e}")
                    break
        
        return downloaded
    
    def _save_metadata(self):
        """メタデータ JSON 保存"""
        
        metadata_file = self.output_dir / "metadata.json"
        metadata_dict = {
            "total_images": len(self.metadata),
            "styles": list(self.STYLE_TAGS.keys()),
            "training_data": self.metadata
        }
        
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f, ensure_ascii=False, indent=2)
        
        # ログファイル保存
        log_file = self.output_dir / "download_log.txt"
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("\n".join(self.download_log))
        
        print(f"\n📊 Metadata saved: {metadata_file}")
        print(f"📋 Log saved: {log_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Danbooru から画像をダウンロード")
    parser.add_argument("--output", default="training_data", help="出力ディレクトリ")
    parser.add_argument("--limit", type=int, default=60, help="スタイルあたりの枚数")
    
    args = parser.parse_args()
    
    downloader = DanbooruDownloader(output_dir=args.output)
    total = downloader.download_images(limit_per_style=args.limit)
    
    print(f"\n🎉 完了: {total} 枚の画像を収集しました")
```

**実行コマンド:**

```bash
# 実行前に scripts ディレクトリ作成
mkdir -p scripts
touch scripts/download_danbooru.py
# （上記スクリプトをコピペ）

# 実行
python scripts/download_danbooru.py --output training_data --limit 60

# 結果確認
ls -lh training_data/
# → impressionist_style/, watercolor_aesthetic/, ... が作成される
```

### 1.3 データセット検証

```python
# scripts/validate_dataset.py

import os
from PIL import Image
from pathlib import Path

def validate_training_data(data_dir="training_data"):
    """学習データの有効性チェック"""
    
    print("📊 Validating training data...\n")
    
    data_path = Path(data_dir)
    total_images = 0
    issues = []
    
    for style_dir in data_path.iterdir():
        if not style_dir.is_dir() or style_dir.name.startswith("."):
            continue
        
        print(f"📁 {style_dir.name}/")
        style_count = 0
        
        for img_file in style_dir.glob("*.png"):
            try:
                img = Image.open(img_file)
                # 推奨: 512x512 付近
                if img.size[0] < 256 or img.size[1] < 256:
                    issues.append(f"⚠️  Small image: {img_file.name} ({img.size})")
                style_count += 1
            except Exception as e:
                issues.append(f"❌ Corrupt: {img_file.name} - {e}")
        
        print(f"   ✅ {style_count} images valid")
        total_images += style_count
    
    print(f"\n📊 Summary:")
    print(f"   Total images: {total_images}")
    print(f"   Issues found: {len(issues)}")
    
    if issues:
        print("\n⚠️  Issues:")
        for issue in issues[:10]:  # 最初の10件表示
            print(f"   {issue}")
    
    return total_images >= 200  # 最低200枚必要

if __name__ == "__main__":
    is_valid = validate_training_data()
    print(f"\n{'✅ Ready' if is_valid else '❌ Needs more data'}")
```

---

## 🧠 Step 2: LoRA 学習スクリプト設計

### 2.1 train_lora.py の設計方針

**ファイル: `train_lora.py`** (実装は次項)

```python
"""
Anime Impressionist LoRA Training Script

特徴:
- PEFT (Parameter-Efficient Fine-Tuning) 使用
- Dreambooth 学習テクニック組み込み
- 学習進度の可視化
- チェックポイント保存機能
"""

class LoRATrainer:
    def __init__(self, training_data_dir, output_dir):
        # Stable Diffusion v1.5 をロード
        # LoRA コンフィグ設定
        pass
    
    def train(self, num_train_epochs=50, learning_rate=1e-4):
        # 学習ループ
        # 損失関数 + 最適化
        pass
    
    def save_lora_weights(self, save_path):
        # LoRA 重みのみ保存 (~4MB)
        pass
```

### 2.2 ハイパーパラメータ設定

```
学習設定:
├─ Model: Stable Diffusion v1.5
├─ Learning Rate: 1e-4 （やや低め）
├─ Batch Size: 4 (T4 GPU メモリ制約)
├─ Steps: 50-100 epochs
├─ LoRA Rank (r): 8 (デフォルト)
├─ LoRA Alpha (α): 32
├─ Dropout: 0.1
└─ Target Modules: ["to_k", "to_v", "to_q"]

理由:
- 低学習率: 元モデルへの差分を小さく保つ
- Batch Size 4: T4 (16GB) の制約
- Rank 8: 質と効率のバランス
```

---

## 🔧 Step 3: 学習実行（Google Colab）

### 3.1 Colab ノートブック動作手順

```python
# Step 1: 環境構築
!pip install -q peft diffusers transformers accelerate safetensors pillow tqdm

# Step 2: 学習データをアップロード
# → /content/training_data/ に配置

# Step 3: train_lora.py を実行
!python train_lora.py \
    --data_dir training_data \
    --output_dir lora_weights \
    --epochs 50

# Step 4: 出力確認
!ls -lh lora_weights/
# → anime-impressionist-lora.safetensors (4MB 程度)

# Step 5: ローカルにダウンロード
# → Google Drive 経由で保存
```

**推定実行時間:**
- 初回セットアップ: 3-5分
- 学習: 1-2時間
- 総計: 1.5-2.5時間

---

## 🎯 Step 4: 推論統合

### 4.1 character_generator.py への統合

修正位置: `generate_image()` メソッド内

```python
def generate_image(self, prompt, use_lora=False):
    """
    Args:
        prompt: 生成プロンプト
        use_lora: LoRA 重みを適用するか
    """
    
    if use_lora:
        self.pipe.load_lora_weights("lora_weights/anime-impressionist-lora.safetensors")
        print("📚 LoRA weights loaded")
    
    image = self.pipe(
        prompt=prompt,
        negative_prompt="low quality",
        num_inference_steps=20,
        guidance_scale=7.0
    ).images[0]
    
    if use_lora:
        self.pipe.unload_lora_weights()
    
    return image
```

### 4.2 テスト実行

```python
generator = AnimeCharacterGenerator()

# v1.5 デフォルト
img1 = generator.generate_image("1girl, watercolor style")

# LoRA 適用版
img2 = generator.generate_image("1girl, watercolor style", use_lora=True)

# 比較表示
print("v1.5 vs LoRA 適用版の差分を視覚的に確認")
```

---

## 📝 Step 5: ドキュメント化

### 5.1 ブログ記事アウトライン

**記事タイトル:**  
「LoRA ファインチューニングで生成 AI を『印象派風アニメ』に特化させる」

**構成:**

```
1. なぜ LoRA が必要か
   - 汎用モデルの限界
   - ファインチューニングの種類（Full vs LoRA）
   - LoRA の利点（軽量、高速）

2. データセット準備の工夫
   - Danbooru タグ戦略
   - 印象派風画像の特徴
   - metadata.json 構造

3. LoRA 学習の深掘り
   - PEFT ライブラリの仕組み
   - ハイパーパラメータ解説
   - 学習曲線の読み方

4. 実装と推論
   - 学習スクリプト解説
   - Colab での実行手順
   - 推論時間の測定

5. 結果比較
   - v1.5 vs LoRA図鑑
   - 品質評価
   - 失敗事例と対策

6. 今後の改善
   - 複数 LoRA の組み合わせ
   - Controlnet との統合
```

### 5.2 README.md 更新

```markdown
## 🎨 LoRA ファインチューニング対応

v1.0 から Phase 2 へ進化。独自スタイル（印象派風）にファインチューニング。

### 使用方法

\`\`\`python
generator = AnimeCharacterGenerator()

# LoRA 適用
image = generator.generate_image(
    prompt="1girl, masterpiece",
    use_lora=True
)
\`\`\`

### LoRA モデル情報

- **ダウンロード**: [HuggingFace Hub](https://huggingface.co/Shion1124/anime-impressionist-lora)
- **サイズ**: 4MB
- **学習データ**: 300 images (Impressionist style)
```

---

## 📊 Step 6: 本番化

### 6.1 HuggingFace Hub アップロード

```bash
# 1. モデルカード作成 (README.md 形式)
cat > lora_weights/README.md << 'EOF'
# Anime Impressionist LoRA

Stable Diffusion v1.5 向けのカスタム LoRA ウェイト

## 学習データ
- スタイル: 印象派的、水彩画的
- 枚数: 300 images
- ポジティブタグ: watercolor, impressionist, soft focus, aesthetic

## 使用例
```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.load_lora_weights("Shion1124/anime-impressionist-lora")
```
EOF

# 2. HuggingFace にアップロード
huggingface-cli repo create anime-impressionist-lora --type model
cd lora_weights
git clone https://huggingface.co/Shion1124/anime-impressionist-lora
cd anime-impressionist-lora
cp ../anime-impressionist-lora.safetensors .
cp ../README.md .
git add .
git commit -m "Add LoRA weights"
git push
```

### 6.2 GitHub に反映

```bash
cd /path/to/anime-character-generator

# ファイル追加
git add train_lora.py
git add scripts/download_danbooru.py
git add dev_peft.md
git add lora_weights/anime-impressionist-lora.safetensors

# コミット
git commit -m "Phase 2: Add LoRA fine-tuning implementation

- Add Danbooru downloader script
- Implement LoRA training pipeline
- Add trained model weights (~4MB)
- Update documentation"

git push origin master
```

---

## ⏱️ タイムライン （推定）

| Week | Task | 日数 | 完了条件 |
|------|------|------|--------|
| 1 | データセット準備 | 2日 | 300 枚 + metadata.json ✅ |
| 1 | 学習スクリプト実装 | 2日 | train_lora.py 完成 ✅ |
| 1 | 学習実行 | 1日 | .safetensors 生成 ✅ |
| 2 | 推論統合 | 1日 | character_generator.py 更新 ✅ |
| 2 | テスト + 品質評価 | 1日 | 比較画像作成 ✅ |
| 2 | ドキュメント化 | 1.5日 | ブログ記事完成 ✅ |
| 2 | 本番化 | 0.5日 | HuggingFace Hub 公開 ✅ |

**合計: 7-10日**

---

## 🎯 Success Criteria（成功指標）

```
✅ データセット
   - 300 枚以上の高品質アニメ画像を収集
   - metadata.json で適切にタグ付け

✅ モデル学習
   - 学習損失が収束（最終損失 < 0.1）
   - 推論時間 3-5 秒 / 画像以内

✅ 品質評価
   - v1.5 比で「印象派的」要素 + 20%
   - ユーザー評価スコア 8/10 以上

✅ ドキュメント
   - ブログ記事 3000+ 単語
   - HuggingFace Hub で公開

✅ コミュニティ
   - GitHub Star 数向上
   - ダウンロード数が増加
```

---

## 🔗 参考資料

- **PEFT ライブラリ**: https://github.com/huggingface/peft
- **Diffusers LoRA ガイド**: https://huggingface.co/docs/diffusers/training/lora
- **Danbooru API**: https://danbooru.donmai.us/wiki_pages/api
- **Dreambooth 論文**: https://arxiv.org/abs/2208.12242

---

## 📌 注意事項

```
⚠️  ライセンス確認
   - Danbooru からのダウンロード画像の利用規約確認

⚠️  GPU リソース
   - Colab T4 で推奨（無料アカウントで OK）
   - 学習中は セルを閉じない

⚠️  データプライバシー
   - 収集した画像は HuggingFace Hub 公開時に確認

⚠️  著作権
   - 二次利用可能な画像のみ学習データに含める
```

---

**作成日**: 2026年2月18日  
**バージョン**: 1.0  
**ステータス**: 計画段階 → 準備段階へ移行予定
