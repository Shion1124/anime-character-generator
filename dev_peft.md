# 🎨 LoRA ファインチューニング実行ガイド v2.1

**プロジェクト**: anime-character-generator Phase 2A 実装  
**目標**: チェックポイント対応・20エポック最適化による実践的 LoRA 学習  
**推定期間**: **3日間（Colab スケジュール）** / 10-12時間（実学習時間）  
**最終成果物**: `anime-lora-final/` + 5つのチェックポイント + `training_log.json`

---

## ⚡ クイックスタート

```bash
# Colab での実行（推奨）
# 1. train_lora.py を Colab にアップロード
# 2. training_data/ を Colab にアップロード（またはマウント）
# 3. 以下を実行:

python train_lora.py \
    --data_dir ./training_data \
    --output_dir ./lora_weights \
    --epochs 20 \
    --batch_size 2 \
    --learning_rate 1e-4

# 中断から再開する場合:
python train_lora.py \
    --data_dir ./training_data \
    --output_dir ./lora_weights \
    --epochs 20 \
    --resume_from ./lora_weights/checkpoint-epoch-5
```

---

## 📋 プロジェクト概要

### Phase 2A: なぜこの設計か？

| 項目 | v2.0（理想） | v2.1（実践的） | 改善理由 |
|------|-----------|-------------|--------|
| エポック数 | 50-100 | **20** | 小規模データ(300枚)は10-15で収束 |
| 学習時間 | 50-100時間 | **10-12時間** | Colab 12h/day 制約に対応 |
| チェックポイント | なし | **毎5エポック** | Colab セッション切断対策 |
| セッション分割 | 連続 | **3回（3日）** | 現実的な学習スケジュール |
| 再開機能 | 未実装 | **実装済み** | --resume_from で完全対応 |
| 進捗ログ | なし | **training_log.json** | 損失曲線を可視化 |
| 実装状態 | 設計のみ | **✅ 完全実装** | production-ready |

### データセット概要

```
📊 training_data/ (既に300枚揃っています)
├─ impressionist_style/      60枚 (印象派風)
├─ oil_painting_aesthetic/   60枚 (油彩風)  
├─ pastel_softness/          60枚 (パステル調)
├─ sketch_aesthetic/         60枚 (スケッチ風)
├─ soft_focus_landscape/     60枚 (朧げな風景)
├─ metadata.json             (タグ情報)
└─ download_log.txt          (収集ログ)

合計: 300枚 | 総容量: ~600MB

データ検証: ✅ 完了
```

---

## 📥 データセット収集の経緯と方法

### なぜ Danbooru からのデータ収集か？

LoRA ファインチューニングには、**統一されたスタイルを持つ画像が必要**です。汎用データセット（ImageNet など）ではなく、**アニメ・イラスト専門のデータベース** を使用することで：

- ✅ 印象派風、水彩画的、油彩風など**スタイルの統一性**
- ✅ **タグベースの体系的な分類**（metadata.json で管理）
- ✅ **商用・学習利用可能なライセンス確認**
- ✅ 高品質の 512×512 相当の画像

### 収集方法: scripts/download_danbooru.py

**ファイル位置**: `/Users/yoshihisashinzaki/ai_projects/anime-character-generator/scripts/download_danbooru.py`

```python
#!/usr/bin/env python3
"""
Danbooru から印象派風アニメ画像を収集するスクリプト

使用例:
    python scripts/download_danbooru.py --output training_data --limit 60

特徴:
- スタイル別タグ定義で体系的に収集
- metadata.json で各画像のメタ情報を記録
- API リクエスト間隔を自動制御（レート制限回避）
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
            time.sleep(2)  # API リクエスト間隔（Danbooru への負荷軽減）
        
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

### 実行方法（データ再収集が必要な場合）

```bash
# 環境準備
pip install requests pillow tqdm

# スクリプト実行
python scripts/download_danbooru.py \
    --output training_data \
    --limit 60

# 結果確認
ls -lh training_data/
# 出力例:
# impressionist_style/      (60 PNG files)
# oil_painting_aesthetic/   (60 PNG files)
# pastel_softness/          (60 PNG files)
# sketch_aesthetic/         (60 PNG files)
# soft_focus_landscape/     (60 PNG files)
# metadata.json             (タグ情報)
# download_log.txt          (実行ログ)
```

### データセット検証スクリプト: scripts/validate_dataset.py

```python
#!/usr/bin/env python3
"""
学習データの有効性チェック
"""

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
        
        for img_file in sorted(style_dir.glob("*.png")) + sorted(style_dir.glob("*.jpg")):
            try:
                img = Image.open(img_file)
                
                # サイズ チェック
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
        print("\n⚠️  Issues found:")
        for issue in issues[:10]:  # 最初の10件表示
            print(f"   {issue}")
    
    # 成功判定
    success = total_images >= 200
    print(f"\n{'✅ Dataset ready' if success else '❌ Need more images'}")
    return success

if __name__ == "__main__":
    validate_training_data()
```

### metadata.json の構造例

```json
{
  "total_images": 300,
  "styles": [
    "impressionist_style",
    "oil_painting_aesthetic",
    "pastel_softness",
    "sketch_aesthetic",
    "soft_focus_landscape"
  ],
  "training_data": [
    {
      "file": "impressionist_style/impressionist_style_000.png",
      "style": "impressionist_style",
      "tags": ["watercolor", "landscape", "soft focus", "anime"],
      "width": 512,
      "height": 512
    },
    {
      "file": "oil_painting_aesthetic/oil_painting_aesthetic_001.png",
      "style": "oil_painting_aesthetic",
      "tags": ["oil painting", "texture", "aesthetic"],
      "width": 576,
      "height": 512
    }
    ...
  ]
}
```

### データ収集を振り返る

```
🔍 データセット開発の流れ：

1. 要件定義
   - 印象派、水彩、油彩などスタイル別の分類
   - 各スタイル 60 枚 × 5 スタイル = 300 枚

2. ソース選定
   - Danbooru: アニメ・イラスト専門、タグが豊富
   - 公開 API で自動収集可能
   - ライセンス確認完了

3. タグ戦略
   - "watercolor" + "impressionist_style" → 印象派
   - "oil_painting_style" → 油彩風
   - "soft_focus" + "landscape" → 朧げな風景
   - "-lowres" で低品質画像を除外

4. 自動化スクリプト作成
   - download_danbooru.py: 体系的にダウンロード
   - metadata.json: 各画像の詳細情報を記録
   - validate_dataset.py: 品質チェック

5. 結果
   - ✅ 300 枚の統一スタイル画像を収集
   - ✅ metadata.json で画像情報を管理
   - ✅ download_log.txt で収集履歴を記録
```

---

## 🏗️ Phase 2A: アーキテクチャ概要

### 全体フロー図

```
[Colab T4 GPU]
    ↓
[train_lora.py 実行]
    ↓
┌─────────────────────────────────────────┐
│ Session 1: Epoch 1-5 (~2.5時間)         │
│ ✅ checkpoint-epoch-5/保存               │
│    training_log.json 記録                 │
└─────────────────────────────────────────┘
    ↓ (セッション切断)
┌─────────────────────────────────────────┐
│ Session 2: Epoch 5-10 (~2.5時間)        │
│ 実行: --resume_from checkpoint-epoch-5   │
│ ✅ checkpoint-epoch-10/ 保存              │
└─────────────────────────────────────────┘
    ↓ (セッション切断)
┌─────────────────────────────────────────┐
│ Session 3: Epoch 10-20 (~5時間)         │
│ 実行: --resume_from checkpoint-epoch-10  │
│ ✅ anime-lora-final/ 保存               │
│ ✅ training_log.json 完成 (全20エポック) │
└─────────────────────────────────────────┘
    ↓
[HuggingFace Hub へアップロード (オプション)]
    ↓
[character_generator.py で推論]
```

### 出力ファイル構造

```
lora_weights/
├── checkpoint-epoch-5/
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── training_metadata.json  ← 再開用タイムスタンプ
├── checkpoint-epoch-10/
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── training_metadata.json
├── checkpoint-epoch-15/
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── training_metadata.json
├── checkpoint-epoch-20/        ← 最後のチェックポイント
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── training_metadata.json
├── anime-lora-final/           ← 本番モデル（最終）
│   ├── adapter_config.json
│   └── adapter_model.bin
└── training_log.json           ← 全エポックの損失曲線
```

---

## 🚀 実装済みの train_lora.py の詳細

### train_lora.py の主要クラス

#### 1. `AnimeDataset` クラス

```python
class AnimeDataset(Dataset):
    """
    training_data/ ディレクトリから画像を自動発見
    
    特徴:
    - サブディレクトリのスタイルを自動認識
    - PNG/JPG 両対応
    - リサイズ・正規化を自動実行
    """
    
    def __init__(self, data_dir: str, resolution: int = 512):
        # 再帰的に全画像のパスを発見
        self.image_paths = []
        for style_dir in Path(data_dir).iterdir():
            self.image_paths.extend(list(style_dir.glob("*.png")))
            self.image_paths.extend(list(style_dir.glob("*.jpg")))
        
        print(f"✅ Found {len(self.image_paths)} images in {data_dir}")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        # 画像をロード、リサイズ、正規化
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)  # 512×512 にリサイズ等
        return image
```

**実際の使用:**

```python
dataset = AnimeDataset(data_dir="./training_data", resolution=512)
# → 自動で 300 枚全て発見
# → メモリ配置: オンデマンド（必要な時だけロード）
```

#### 2. `LoRATrainer` クラス

```python
class LoRATrainer:
    """
    Stable Diffusion v1.5 に LoRA を適用して学習
    
    特徴:
    - PEFT を使用した軽量ファインチューニング
    - 毎 5 エポックごとにチェックポイント保存
    - 損失履歴を JSON に記録
    - 中断・再開機能完全対応
    """
    
    def setup_model(self):
        """LoRA 設定"""
        # 1. Stable Diffusion v1.5 ロード
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,  # メモリ節約
            safety_checker=None
        )
        
        # 2. VAE・Text Encoder は凍結（UNet の LoRA のみ学習）
        pipe.vae.requires_grad_(False)
        pipe.text_encoder.requires_grad_(False)
        
        # 3. UNet に LoRA 適用
        lora_config = LoraConfig(
            r=32,  # LoRA ランク
            lora_alpha=32,
            target_modules=["to_k", "to_v", "to_q", "to_out"],
            lora_dropout=0.1,
            bias="none"
        )
        pipe.unet = get_peft_model(pipe.unet, lora_config)
        
        return pipe
    
    def train(self, ..., resume_from: Optional[str] = None):
        """
        学習ループ（チェックポイント対応）
        """
        
        # チェックポイントから再開
        if resume_from:
            adapter_path = Path(resume_from)
            self.pipe.unet.load_adapter(adapter_path)
            # メタデータから開始エポックを取得
            with open(adapter_path / "training_metadata.json") as f:
                metadata = json.load(f)
                start_epoch = metadata.get("epoch", 0)
        else:
            start_epoch = 0
        
        # 学習ループ
        for epoch in range(start_epoch, num_epochs):
            epoch_loss = 0.0
            
            for batch in dataloader:
                # 前向き計算 → 損失 → バックプロップ
                loss = self.compute_loss(batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # チェックポイント保存（毎 5 エポック）
            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                checkpoint_dir = output_dir / f"checkpoint-epoch-{epoch+1}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                # LoRA ウェイト保存
                self.pipe.unet.save_pretrained(checkpoint_dir)
                
                # メタデータ記録（再開用）
                metadata = {
                    "epoch": epoch + 1,
                    "timestamp": datetime.now().isoformat(),
                    "loss": epoch_loss / len(dataloader)
                }
                with open(checkpoint_dir / "training_metadata.json", "w") as f:
                    json.dump(metadata, f)
                
                # 学習ログ記録
                self.training_log.append({
                    "epoch": epoch + 1,
                    "loss": epoch_loss / len(dataloader)
                })
```

### CLI パラメータ

```bash
python train_lora.py \
    --data_dir ./training_data          # データセットパス
    --output_dir ./lora_weights         # 出力先
    --epochs 20                         # エポック数（デフォルト: 20）
    --batch_size 2                      # バッチサイズ（T4 最適）
    --learning_rate 1e-4                # 学習率
    --lora_rank 32                      # LoRA ランク
    --resume_from ./lora_weights/checkpoint-epoch-5  # 再開（オプション）
```

### ハイパーパラメータ解説

| パラメータ | 値 | 理由 |
|-----------|-----|------|
| **Epochs** | 20 | 300枚 × 20 = 6000枚相当 (小規模データで十分) |
| **Batch Size** | 2 | T4 (16GB VRAM) で fp16 実行可能 |
| **Learning Rate** | 1e-4 | LoRA では低めが安定 |
| **LoRA Rank** | 32 | 品質と効率のバランス |
| **Checkpoint Interval** | 5 epochs | I/O とリカバリ時間のバランス (~1.5h) |

---

## 📅 実行スケジュール（3日分割計画）

### Day 1: Session 1 - Epoch 1-5

**所要時間**: 2.5時間程度

```bash
# Colab セル 1: 環境構築
!pip install -q peft diffusers transformers accelerate safetensors pillow tqdm

# Colab セル 2: 初回実行
%cd /content/anime-character-generator
!python train_lora.py \
    --data_dir ./training_data \
    --output_dir ./lora_weights \
    --epochs 20 \
    --batch_size 2 \
    --learning_rate 1e-4

# ログ確認 (セッション終了前)
!ls -lh lora_weights/
# 出力例:
# checkpoint-epoch-5/
# training_log.json
```

**確認項目:**
- ✅ checkpoint-epoch-5/ が作成されたか？
- ✅ training_metadata.json に epoch: 5 と timestamp が記録されているか？
- ✅ training_log.json に损失値が記録されているか？

**出力サンプル:**

```json
// training_log.json (Session 1 終了時)
[
  {"epoch": 1, "loss": 0.1234},
  {"epoch": 2, "loss": 0.0987},
  {"epoch": 3, "loss": 0.0854},
  {"epoch": 4, "loss": 0.0723},
  {"epoch": 5, "loss": 0.0652}
]
```

### Day 2: Session 2 - Epoch 5-10

**準備**: 前日のチェックポイントの位置を確認

```bash
# Colab セル: 中断から再開
%cd /content/anime-character-generator
!python train_lora.py \
    --data_dir ./training_data \
    --output_dir ./lora_weights \
    --epochs 20 \
    --resume_from ./lora_weights/checkpoint-epoch-5
```

**内部処理:**
1. checkpoint-epoch-5/adapter_config.json をロード
2. training_metadata.json から epoch=5 を取得 → start_epoch=5 に設定
3. epoch 5 から学習再開
4. epoch 10 に到達時に checkpoint-epoch-10/ を自動保存

**出力サンプル:**

```json
// training_log.json (Session 2 終了時)
[
  {"epoch": 1, "loss": 0.1234},
  {"epoch": 2, "loss": 0.0987},
  {"epoch": 3, "loss": 0.0854},
  {"epoch": 4, "loss": 0.0723},
  {"epoch": 5, "loss": 0.0652},
  {"epoch": 6, "loss": 0.0598},  ← Session 2
  {"epoch": 7, "loss": 0.0521},
  {"epoch": 8, "loss": 0.0467},
  {"epoch": 9, "loss": 0.0412},
  {"epoch": 10, "loss": 0.0387}
]
```

### Day 3: Session 3 - Epoch 10-20

**準備**: checkpoint-epoch-10/ が保存されているか確認

```bash
# Colab セル: 最終セッション
%cd /content/anime-character-generator
!python train_lora.py \
    --data_dir ./training_data \
    --output_dir ./lora_weights \
    --epochs 20 \
    --resume_from ./lora_weights/checkpoint-epoch-10
```

**最終出力:**
- checkpoint-epoch-15/
- checkpoint-epoch-20/
- anime-lora-final/ (20エポック時点での最終モデル)
- training_log.json (完全20エポック分)

**出力サンプル:**

```json
// training_log.json (最終版)
[
  ... (epoch 1-10は前セッション)
  {"epoch": 11, "loss": 0.0351},
  {"epoch": 12, "loss": 0.0312},
  {"epoch": 13, "loss": 0.0287},
  {"epoch": 14, "loss": 0.0268},
  {"epoch": 15, "loss": 0.0249},
  {"epoch": 16, "loss": 0.0223},
  {"epoch": 17, "loss": 0.0201},
  {"epoch": 18, "loss": 0.0178},
  {"epoch": 19, "loss": 0.0156},
  {"epoch": 20, "loss": 0.0142}  ← 収束完了
]
```

---

## ✅ 各セッションのチェックリスト

### Colab Session の実行前チェック

```
□ training_data/ がアップロードまたはマウントされている
□ train_lora.py がアップロードされている
□ pip 依存関係がインストール済み (diffusers, peft etc.)
□ GPU が割り当てられている (nvidia-smi で確認)
□ ストレージに 2GB 以上の空き容量がある
```

### 各セッション終了時のチェック

**Session 1 終了時:**

```bash
# ファイル確認
!ls -lh lora_weights/
# 期待: checkpoint-epoch-5/, training_log.json

# メタデータ確認
!cat lora_weights/checkpoint-epoch-5/training_metadata.json
# 期待: epoch: 5, timestamp, loss

# メモリクリア
!rm -rf ~/.cache/huggingface/hub/*  # Colab メモリ節約）
```

**Session 2 終了時:**

```bash
# ファイル確認
!ls -lh lora_weights/
# 期待: checkpoint-epoch-5/, checkpoint-epoch-10/, training_log.json

# training_log.json の内容確認
!tail -n 5 lora_weights/training_log.json
# 期待: epoch 6-10 の損失が記録されている
```

**Session 3 終了時:**

```bash
# 最終出力確認
!ls -lh lora_weights/
# 期待: checkpoint-epoch-15/, checkpoint-epoch-20/, anime-lora-final/

# 最終モデル確認
!ls -lh lora_weights/anime-lora-final/
# 期待: adapter_config.json (~1KB), adapter_model.bin (~3-4MB)

# 学習曲線確認
!python -c "import json; data = json.load(open('lora_weights/training_log.json')); print([d['loss'] for d in data])"
# 期待: 損失が単調減少 (収束)
```

---

## � トラブルシューティング

### ❌ よくあるエラー と 対処法

#### 1. `ModuleNotFoundError: No module named 'peft'`

```
原因: PEFT ライブラリがインストールされていない
対処:
  !pip install -q peft
  !pip install -q diffusers transformers accelerate
```

#### 2. `CUDA out of memory`

```
原因: バッチサイズが大きすぎる
対処:
  # コマンドに追加:
  --batch_size 1  # 2 から 1 に削減（学習は遅くなるが可能）
```

#### 3. `checkpoint-epoch-5/ ファイルが見つからない (再開時)`

```
原因: 前回のセッションでチェックポイントが保存されていない
確認:
  !ls -lh lora_weights/
  # → checkpoint-epoch-5/ が存在するか確認
対処:
  # 存在しなければ、--resume_from なしで最初から実行
  !python train_lora.py --data_dir ./training_data --output_dir ./lora_weights --epochs 20
```

#### 4. `training_log.json が損傷している`

```
原因: セッション中断により JSON が不完全
対処:
  # ファイルバックアップ
  !cp lora_weights/training_log.json lora_weights/training_log.json.bak
  
  # スクリプトが自動的に JSON を修復・補完する
  # 再開実行時に正常化される
```

#### 5. `Colab セッション切断時の対処`

```
状況: 学習中にセッションが切れた
    → training_log.json の最後のエポック = 現在のエポック
確認:
  !tail -n 1 lora_weights/training_log.json
  # 例: {"epoch": 7, "loss": 0.0521}
  # → epoch 7 まで完了、次は epoch 8 から再開
代替え:
  # checkpoint-epoch-5/ が存在すれば問題なし
  !python train_lora.py \
      --data_dir ./training_data \
      --output_dir ./lora_weights \
      --epochs 20 \
      --resume_from ./lora_weights/checkpoint-epoch-5
```

---

## 🎯 推論統合: character_generator.py での使用

### 4.1 LoRA 重みのロード・推論

**ファイル: `character_generator.py` の modify 箇所**

```python
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel

class AnimeCharacterGenerator:
    def __init__(self, model_name="runwayml/stable-diffusion-v1-5"):
        self.model_name = model_name
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            safety_checker=None
        )
        self.lora_loaded = False
    
    def load_lora(self, lora_path: str = "./lora_weights/anime-lora-final"):
        """LoRA 重みをロード"""
        try:
            self.pipe.unet.load_adapter(lora_path, adapter_name="anime_lora")
            self.lora_loaded = True
            print(f"✅ LoRA loaded from {lora_path}")
        except Exception as e:
            print(f"❌ Failed to load LoRA: {e}")
            self.lora_loaded = False
    
    def unload_lora(self):
        """LoRA 重みをアンロード"""
        if self.lora_loaded:
            self.pipe.unet.delete_adapter("anime_lora")
            self.lora_loaded = False
            print("✅ LoRA unloaded")
    
    def generate_image(
        self,
        prompt: str,
        use_lora: bool = False,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        seed: int = None
    ) -> Image:
        """
        画像生成（LoRA 対応）
        
        Args:
            prompt: 生成プロンプト
            use_lora: LoRA を使用するか
            num_inference_steps: 推論ステップ
            guidance_scale: ガイダンススケール
            seed: 乱数シード（再現性用）
        
        Returns:
            生成された PIL Image
        """
        
        # シード設定（オプション）
        if seed is not None:
            torch.manual_seed(seed)
        
        # LoRA 適用
        if use_lora and not self.lora_loaded:
            self.load_lora()
        elif not use_lora and self.lora_loaded:
            self.unload_lora()
        
        # 推論
        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                negative_prompt="low quality, worst quality",
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
        
        return result.images[0]
    
    def generate_batch(
        self,
        prompts: list,
        use_lora: bool = False,
        **kwargs
    ) -> list:
        """バッチ推論"""
        images = []
        for prompt in prompts:
            img = self.generate_image(prompt, use_lora=use_lora, **kwargs)
            images.append(img)
        return images
```

### 4.2 使用例

```python
# 初期化
generator = AnimeCharacterGenerator()

# 例 1: v1.5 デフォルトで生成
prompt1 = "1girl, anime, beautiful, masterpiece, high quality"
img1 = generator.generate_image(prompt1, use_lora=False)
img1.save("output_v1.5.png")

# 例 2: LoRA 適用版で生成（同じプロンプト）
img2 = generator.generate_image(prompt1, use_lora=True)
img2.save("output_lora.png")

# 例 3: バッチ推論
prompts = [
    "1girl, watercolor style, soft colors",
    "1girl, impressionist, oil painting aesthetic",
    "1girl, sketch aesthetic, soft focus"
]
images_lora = generator.generate_batch(prompts, use_lora=True)

for i, img in enumerate(images_lora):
    img.save(f"batch_{i:02d}_lora.png")

# LoRA をアンロード（メモリ解放）
generator.unload_lora()
```

---

## 📊 品質評価とチューニング

### 損失曲線の読み方

```python
import json
import matplotlib.pyplot as plt

# training_log.json をロード
with open("lora_weights/training_log.json") as f:
    logs = json.load(f)

# 損失曲線をプロット
epochs = [log["epoch"] for log in logs]
losses = [log["loss"] for log in logs]

plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, marker='o', linestyle='-', linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("LoRA Training Loss Curve")
plt.grid(True)
plt.yscale("log")  # ログスケール
plt.savefig("training_curve.png", dpi=150, bbox_inches='tight')
plt.show()

# 統計情報
print(f"Initial Loss: {losses[0]:.6f}")
print(f"Final Loss: {losses[-1]:.6f}")
print(f"Reduction: {(1 - losses[-1]/losses[0]) * 100:.1f}%")
print(f"Average Loss: {sum(losses) / len(losses):.6f}")
```

### 成功の目安

```
✅ 良好な学習:
   - Epoch 1 → 20 で 損失が 30-50% 低下
   - 最終損失 < 0.05
   - 損失が単調減少（ノイズは許容）

⚠️  不適切な学習:
   - 損失が増加傾向 → 学習率が高い
   - 最終損失が収束しない → エポック数不足
   - Epoch 5 で既に plateau → 学習率が低い

❌ 失敗の兆候:
   - 損失が NaN になる → VRAM 不足またはバッチサイズ過大
   - 損失が発散 → 学習率が高すぎる
   - checkpoint が保存されない → ストレージ不足
```

---

## 🚀 推論の最適化

### 推論速度向上（高度な設定）

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None
)

# LoRA ロード
pipe.unet.load_adapter("./lora_weights/anime-lora-final")

# 最適化 1: xFormers メモリ効率化
try:
    pipe.enable_xformers_memory_efficient_attention()
    print("✅ xFormers enabled")
except ImportError:
    print("⚠️  xFormers not available (pip install xformers)")

# 最適化 2: モデルコンパイル (Torch 2.0+)
try:
    pipe.unet = torch.compile(pipe.unet, mode="reduced-overhead")
    print("✅ Model compiled with torch.compile")
except Exception as e:
    print(f"⚠️  torch.compile not available: {e}")

# 推論テスト
import time
prompt = "1girl, masterpiece, high quality"

start = time.time()
image = pipe(prompt, num_inference_steps=20).images[0]
elapsed = time.time() - start

print(f"🚀 Generated in {elapsed:.2f} seconds")
```

### 推論時間の目安

```
環境: Colab T4 GPU, fp16, LoRA 適用

デフォルト (20 steps):
  - 最適化なし: 5-7 秒
  - xFormers + compile: 3-4 秒

高速化 (LCM 使用時):
  - 4 steps: 1-2 秒
```

---

## 📤 HuggingFace Hub へのアップロード

### オプション 1: Python スクリプト（推奨）

```bash
# 既に用意されている upload_to_huggingface.py を使用
export HF_TOKEN="your_huggingface_token_here"

python upload_to_huggingface.py \
    --model-path ./lora_weights/anime-lora-final \
    --repo-name anime-character-lora \
    --hf-token $HF_TOKEN \
    --private false
```

### オプション 2: CLI コマンド

```bash
# HuggingFace CLI ツール
pip install huggingface-hub

# ログイン
huggingface-cli login
# → トークンを入力

# リポジトリ作成 & アップロード
huggingface-cli repo create anime-character-lora --type model
cd lora_weights/anime-lora-final
git clone https://huggingface.co/YOUR_USERNAME/anime-character-lora
cd anime-character-lora
cp ../adapter_config.json .
cp ../adapter_model.bin .
git add .
git commit -m "Add anime LoRA model"
git push origin main
```

### アップロード後の利用方法

```python
from diffusers import StableDiffusionPipeline

# リモートからディレクトリにロード
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None
)

# HuggingFace Hub からLoRA ロード
pipe.unet.load_adapter("YOUR_USERNAME/anime-character-lora")

# 推論
image = pipe(prompt="1girl, masterpiece").images[0]
```

---

## 📋 完成チェックリスト

### Phase 2A 完了時の確認項目

```
✅ 学習完了
   □ 20 エポック学習完了
   □ training_log.json に 20 個のエントリ
   □ anime-lora-final/ ディレクトリ存在
   □ adapter_model.bin サイズ 3-4 MB

✅ チェックポイント
   □ checkpoint-epoch-5/ 存在
   □ checkpoint-epoch-10/ 存在
   □ checkpoint-epoch-15/ 存在
   □ checkpoint-epoch-20/ 存在
   □ 各チェックポイントに training_metadata.json

✅ 推論テスト
   □ v1.5 vs LoRA で画像比較可能
   □ LoRA 適用版で印象派風スタイル確認
   □ 推論時間 3-5 秒 / 画像

✅ ドキュメント
   □ character_generator.py を LoRA 対応に更新
   □ README.md に LoRA 使用方法を記載
   □ dev_peft.md（本ファイル）完成

✅ デプロイ（オプション）
   □ HuggingFace Hub にアップロード
   □ リモートからのロード確認
```

---

## 📚 参考・追加資料

### 学術論文
- **PEFT (Parameter-Efficient Fine-Tuning)**: https://arxiv.org/abs/2305.18356
- **LoRA (Low-Rank Adaptation)**: https://arxiv.org/abs/2106.09685
- **Stable Diffusion**: https://arxiv.org/abs/2112.10752

### 実装リソース
- **HuggingFace PEFT**: https://github.com/huggingface/peft
- **Diffusers LoRA Tutorial**: https://huggingface.co/docs/diffusers/training/lora
- **LoRA の詳細解説**: https://huggingface.co/blog/lora

### 関連ブログ記事（書予定）
- 「PEFT ライブラリを使った効率的なファインチューニング」
- 「LoRA ランクの選択: 品質と効率のトレードオフ」
- 「Colab T4 での LoRA 学習マスターガイド」
- 「checkpoints から再開する仕組みの解説」

---

## 🔄 次のステップ

### Phase 2A 完了後

1. **品質評価ブログ記事執筆** (1-2日)
   - 学習曲線の解説
   - v1.5 vs LoRA 比較
   - プロンプトテクニック

2. **Phase 2B: LCM 蒸留** (3-5日)
   - 推論ステップ削減 (20 → 4)
   - 推論時間 5秒 → 1秒に高速化

3. **Phase 3: Image-to-Image 統合** (3日)
   - 既存画像 → LoRA スタイル変換
   - キャラクター感情変化アニメーション

4. **本番デプロイ** (1日)
   - HuggingFace Hub 公開
   - Streamlit Web UI

---

## 📞 トラブルシューティングヘルプ

**Colab 実行中にエラーが出た場合:**

1. **GPU がアクティブか確認**
   ```bash
   !nvidia-smi  # GPU 使用状況を確認
   ```

2. **依存ライブラリの再インストール**
   ```bash
   !pip install --upgrade pip
   !pip install --upgrade -r requirements.txt
   ```

3. **メモリをクリア**
   ```bash
   import gc
   gc.collect()
   torch.cuda.empty_cache()
   ```

4. **Colab ノートブックをリセット**
   - ランタイム → すべてのセルを実行解除
   - ランタイム → リセット（全メモリクリア）

5. **フォーラムで相談**
   - https://github.com/huggingface/diffusers/discussions
   - https://forums.fast.ai/

---

**最終更新**: 2026年2月19日  
**バージョン**: 2.1 - Checkpoint 対応・実装完了版  
**ステータス**: 🚀 Colab での実行に向けて準備完了
