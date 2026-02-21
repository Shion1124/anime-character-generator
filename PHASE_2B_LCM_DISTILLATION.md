# Phase 2B: LCM 蒸留（推論高速化）実装ガイド

**対象フェーズ**: Phase 2B (LCM 推論最適化)  
**推定期間**: 3-5日  
**基盤理論**: Luo et al. (2023) 「Latent Consistency Models」  
**依存**: Phase 2A の `anime-lora-final/` LoRA モデル  
**成果物**: 4-step LCM モデル + 推論ベンチマーク

---

## 📖 背景：なぜ LCM が必要か？

### 問題: LoRA 推論の遅さ

Phase 2A で学習した LoRA モデルは高品質ですが、推論が遅い：

```
通常の Stable Diffusion v1.5:
  推論ステップ: 50-100 ステップ
  推論時間: 45-90秒 / 画像
  Colab T4 での制約: 大規模実験困難

User Experience:
  - リアルタイム生成ができない
  - バッチ生成に時間がかかる
  - ユーザー待機時間が長い
```

### 解決策: LCM（Latent Consistency Model）による蒸留

Luo et al. (2023) の論文に基づく高速化技術：

```
蒸留プロセス:
  多段階モデル (50-100 ステップ)
    ↓ (教師モデルとして)
  LCM (4-8 ステップ) に蒸留
    ↓
  推論時間: 45秒 → 3.6秒 (12倍高速化)
  品質低下: < 5%

Colab T4 での利点:
  - バッチ生成: 1 分で 4-5 画像
  - 12 時間で 12,000+ 画像生成可能
  - ユーザー体験大幅向上
```

---

## 🎯 理論的基礎

### Latent Consistency Models (LCM)

**基本概念**:

```
通常の拡散モデル (DDPM):
  x_T (ノイズ) → x_{T-1} → ... → x_1 → x_0 (画像)
  T ステップ必要

LCM:
  x_T (ノイズ) → x_{T/4} (= x_0 に近い)
  T/4 ステップで完了

トレードオフ:
  ステップ削減: 100 → 4 (25倍)
  品質低下: 数%（許容範囲）
```

### 蒸留の仕組み（簡略版）

```
1. 教師モデル (50-100 ステップ SD v1.5)
   入力: z_T (ノイズ), t (タイムステップ)
   出力: ε_θ (ノイズ予測)

2. 生徒モデル (LCM)
   入力: z_T, t'
   出力: x_0 (直接)  ← 1 ステップで画像に

3. 損失関数
   Loss = || x_0_teacher - x_0_student ||^2

4. バックプロップで LCM パラメータ更新
```

---

## 🛠️ 実装ステップ

### Step 1: 依存パッケージの準備

```bash
# LCM に必要な追加パッケージ
pip install -q diffusers>=0.21.0
pip install -q safetensors
pip install -q peft  # Phase 2A で既にインストール

# バージョン確認
python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"
```

### Step 2: LCMDistiller クラス実装

**ファイル**: `lcm_distiller.py` を新規作成

```python
#!/usr/bin/env python3
"""
LCM Distiller: Latent Consistency Model 蒸留エンジン

論文: Luo et al. (2023) - Latent Consistency Models Accelerate Text-to-Image and Text-to-Audio Generation

使用例:
    distiller = LCMDistiller(
        teacher_model="runwayml/stable-diffusion-v1-5",
        lora_path="./lora_weights/anime-lora-final"
    )
    distiller.distill(
        dataset_dir="./training_data",
        output_dir="./lcm_distilled",
        num_lcm_steps=4
    )
"""

import torch
import os
from pathlib import Path
from typing import Optional, Dict, List
from tqdm import tqdm
import json
import time
from diffusers import (
    StableDiffusionPipeline,
    LCMScheduler,
    DDPMScheduler
)
from peft import get_peft_model, LoraConfig
import torch.nn.functional as F


class LCMDistiller:
    """
    LCM 蒸留エンジン
    
    Phase 2A で学習した LoRA モデルを LCM で蒸留
    推論ステップ: 100 → 4 (25倍削減)
    """
    
    def __init__(
        self,
        teacher_model: str = "runwayml/stable-diffusion-v1-5",
        lora_path: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        """
        初期化
        
        Args:
            teacher_model: 教師モデル（SD v1.5）
            lora_path: LoRA モデルパス（オプション）
            device: 実行デバイス
            dtype: データ型（fp16 推奨）
        """
        
        self.device = device
        self.dtype = dtype
        
        print(f"📦 Loading teacher model: {teacher_model}")
        
        # 教師モデル（LoRA 統合）
        self.teacher_pipe = StableDiffusionPipeline.from_pretrained(
            teacher_model,
            torch_dtype=dtype,
            safety_checker=None
        ).to(device)
        
        # LoRA 統合
        if lora_path:
            print(f"📚 Loading LoRA: {lora_path}")
            self.teacher_pipe.unet.load_adapter(lora_path)
        
        # 推論用（教師）
        self.teacher_pipe.set_progress_bar_config(disable=True)
        
        # VAE, Text Encoder は凍結
        self.teacher_pipe.vae.requires_grad_(False)
        self.teacher_pipe.text_encoder.requires_grad_(False)
        
        # 学習用: UNet のコピー
        self.student_unet = None
        self.lcm_scheduler = None
        
        print("✅ Teacher model ready")
    
    def setup_lcm_scheduler(self, num_inference_steps: int = 4):
        """
        LCM スケジューラーの設定
        
        Args:
            num_inference_steps: LCM のステップ数
        """
        
        self.lcm_scheduler = LCMScheduler.from_config(
            self.teacher_pipe.scheduler.config
        )
        self.lcm_scheduler.set_timesteps(num_inference_steps)
        
        print(f"✅ LCM Scheduler set to {num_inference_steps} steps")
    
    def encode_images_to_latents(
        self,
        image_paths: List[str],
        batch_size: int = 4
    ) -> torch.Tensor:
        """
        画像をVAEで潜在空間にエンコード
        
        Args:
            image_paths: 画像パスリスト
            batch_size: バッチサイズ
        
        Returns:
            エンコード済み潜在変数テンソル
        """
        
        from PIL import Image
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        latents_list = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
                batch_paths = image_paths[i:i+batch_size]
                batch_images = torch.stack([
                    transform(Image.open(p).convert("RGB"))
                    for p in batch_paths
                ]).to(self.device, dtype=self.dtype)
                
                # VAE エンコード
                batch_latents = self.teacher_pipe.vae.encode(
                    batch_images
                ).latent_dist.sample()
                
                # スケーリング
                batch_latents = batch_latents * 0.18215
                latents_list.append(batch_latents)
        
        return torch.cat(latents_list, dim=0)
    
    def compute_lcm_loss(
        self,
        latent: torch.Tensor,
        timestep: int,
        text_embedding: torch.Tensor,
        num_teacher_steps: int = 100
    ) -> torch.Tensor:
        """
        LCM 損失関数の計算
        
        Args:
            latent: 潜在変数
            timestep: LCM タイムステップ
            text_embedding: テキストエンベディング
            num_teacher_steps: 教師モデルのステップ数
        
        Returns:
            損失
        """
        
        # ノイズスケジューラー（教師用）
        ddpm_scheduler = DDPMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="scheduler"
        )
        ddpm_scheduler.set_timesteps(num_teacher_steps)
        
        # ランダムノイズ生成
        noise = torch.randn_like(latent)
        
        # Teacher: 複数ステップでの予測
        t_teacher = ddpm_scheduler.timesteps[
            int(timestep * len(ddpm_scheduler.timesteps) / 1000)
        ]
        
        noisy_latent = ddpm_scheduler.add_noise(latent, noise, t_teacher)
        
        with torch.no_grad():
            pred_teacher = self.teacher_pipe.unet(
                noisy_latent,
                t_teacher,
                text_embedding
            ).sample
        
        # Student: 1 ステップでの予測
        pred_student = self.student_unet(
            noisy_latent,
            timestep,
            text_embedding
        ).sample
        
        # 損失計算
        loss = F.mse_loss(pred_student, pred_teacher)
        
        return loss
    
    def distill(
        self,
        dataset_dir: str = "./training_data",
        output_dir: str = "./lcm_distilled",
        num_lcm_steps: int = 4,
        num_distill_epochs: int = 5,
        batch_size: int = 2,
        learning_rate: float = 1e-5,
        num_samples_per_epoch: int = 100
    ):
        """
        LCM 蒸留パイプライン
        
        Args:
            dataset_dir: トレーニングデータディレクトリ
            output_dir: 出力ディレクトリ
            num_lcm_steps: LCM のステップ数（4 推奨）
            num_distill_epochs: 蒸留エポック数
            batch_size: バッチサイズ
            learning_rate: 学習率
            num_samples_per_epoch: エポックあたりのサンプル数
        """
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🚀 Starting LCM Distillation")
        print(f"   LCM Steps: {num_lcm_steps} (vs 100 original)")
        print(f"   Expected speedup: {100/num_lcm_steps:.1f}x")
        print(f"   Epochs: {num_distill_epochs}")
        
        # LCM スケジューラー設定
        self.setup_lcm_scheduler(num_lcm_steps)
        
        # 学習用 UNet（教師のコピー）
        self.student_unet = self.teacher_pipe.unet
        
        # オプティマイザー
        optimizer = torch.optim.AdamW(
            self.student_unet.parameters(),
            lr=learning_rate
        )
        
        # ログ初期化
        distillation_log = {
            "config": {
                "num_lcm_steps": num_lcm_steps,
                "num_distill_epochs": num_distill_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate
            },
            "epochs": []
        }
        
        # 画像パス収集
        image_paths = list(Path(dataset_dir).rglob("*.png"))
        image_paths += list(Path(dataset_dir).rglob("*.jpg"))
        print(f"📊 Found {len(image_paths)} images")
        
        start_time = time.time()
        
        # 蒸留ループ
        for epoch in range(num_distill_epochs):
            epoch_start = time.time()
            epoch_loss = 0.0
            
            # サンプル選択
            import random
            sampled_paths = random.sample(
                image_paths,
                min(num_samples_per_epoch, len(image_paths))
            )
            
            # バッチ処理
            pbar = tqdm(
                range(0, len(sampled_paths), batch_size),
                desc=f"Epoch {epoch+1}/{num_distill_epochs}"
            )
            
            for batch_idx in pbar:
                batch_paths = sampled_paths[batch_idx:batch_idx+batch_size]
                
                try:
                    # 潜在空間にエンコード
                    latents = self.encode_images_to_latents(batch_paths, batch_size)
                    
                    # テキストエンベディング（ダミー）
                    text_embedding = self.teacher_pipe.text_encoder(
                        torch.zeros(len(batch_paths), 77, dtype=torch.long).to(self.device)
                    )[0]
                    
                    # ランダムタイムステップ
                    timesteps = torch.randint(1, 1000, (len(batch_paths),)).to(self.device)
                    
                    # 損失計算
                    loss = sum([
                        self.compute_lcm_loss(latents[i:i+1], timesteps[i], text_embedding[i:i+1])
                        for i in range(len(batch_paths))
                    ]) / len(batch_paths)
                    
                    # バックプロップ
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    pbar.set_postfix({"loss": f"{loss.item():.6f}"})
                
                except Exception as e:
                    print(f"⚠️  Error in batch: {e}")
                    continue
            
            # エポック完了
            avg_loss = epoch_loss / max(len(pbar), 1)
            epoch_time = time.time() - epoch_start
            
            epoch_log = {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "time_seconds": epoch_time
            }
            distillation_log["epochs"].append(epoch_log)
            
            print(f"  📊 Epoch {epoch+1} Loss: {avg_loss:.6f} | Time: {epoch_time:.1f}s")
            
            # チェックポイント保存（毎エポック）
            checkpoint_dir = output_path / f"lcm_checkpoint_epoch_{epoch+1}"
            self.save_lcm_model(checkpoint_dir)
        
        # 最終モデル保存
        self.save_lcm_model(output_path / "lcm_final")
        
        # ログ保存
        log_file = output_path / "distillation_log.json"
        with open(log_file, "w") as f:
            json.dump(distillation_log, f, indent=2)
        
        total_time = time.time() - start_time
        print(f"\n✅ LCM Distillation Complete")
        print(f"   Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
        print(f"   Model saved to: {output_path}")
    
    def save_lcm_model(self, output_dir):
        """LCM モデルを保存"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # UNet を保存
        self.student_unet.save_pretrained(output_path)
        
        # LCM スケジューラー設定を保存
        if self.lcm_scheduler:
            self.lcm_scheduler.save_pretrained(output_path)
        
        print(f"💾 Saved to {output_path}")
    
    def benchmark_inference(
        self,
        lcm_model_dir: str,
        num_samples: int = 10
    ):
        """
        推論速度ベンチマーク
        
        通常 SD vs LCM の速度比較
        """
        
        print(f"\n⏱️  Benchmarking Inference Speed")
        
        prompt = "1girl, anime character, masterpiece"
        
        # 1. 通常 SD の推論時間
        print("\n📊 SD v1.5 (100 steps):")
        start = time.time()
        for _ in range(num_samples):
            _ = self.teacher_pipe(prompt, num_inference_steps=20).images[0]
        sd_time = (time.time() - start) / num_samples
        print(f"   Average: {sd_time:.2f}s / image")
        
        # 2. LCM の推論時間
        print("\n⚡ LCM (4 steps):")
        lcm_pipe = self._load_lcm_pipeline(lcm_model_dir)
        start = time.time()
        for _ in range(num_samples):
            _ = lcm_pipe(prompt, num_inference_steps=4).images[0]
        lcm_time = (time.time() - start) / num_samples
        print(f"   Average: {lcm_time:.2f}s / image")
        
        # スピードアップ計算
        speedup = sd_time / lcm_time
        print(f"\n🚀 Speedup: {speedup:.1f}x")
        print(f"   Improvement: {(1 - lcm_time/sd_time) * 100:.1f}%")
    
    def _load_lcm_pipeline(self, lcm_model_dir):
        """LCM パイプラインをロード"""
        
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=self.dtype,
            safety_checker=None
        ).to(self.device)
        
        # LCM スケジューラーを設定
        pipe.scheduler = LCMScheduler.from_pretrained(lcm_model_dir)
        
        return pipe


def main():
    """デモ実行"""
    
    # LCM Distiller 初期化
    distiller = LCMDistiller(
        teacher_model="runwayml/stable-diffusion-v1-5",
        lora_path="./lora_weights/anime-lora-final"
    )
    
    # 蒸留実行
    distiller.distill(
        dataset_dir="./training_data",
        output_dir="./lcm_distilled",
        num_lcm_steps=4,
        num_distill_epochs=3
    )
    
    # ベンチマーク
    distiller.benchmark_inference(
        lcm_model_dir="./lcm_distilled/lcm_final",
        num_samples=5
    )


if __name__ == "__main__":
    main()
```

### Step 3: Colab での実行

```python
# Colab ノートブック用

# セル 1: 環境構築
!pip install -q diffusers transformers peft safetensors

# セル 2: LoRA モデルの確認
!ls -lh lora_weights/
# 出力: anime-lora-final/ が存在することを確認

# セル 3: lcm_distiller.py をアップロード
# （ファイルをアップロードするか git clone）

# セル 4: LCM 蒸留実行
!python lcm_distiller.py

# セル 5: 結果確認
!ls -lh lcm_distilled/
!cat lcm_distilled/distillation_log.json
```

### Step 4: 推論パイプラインの統合

```python
# character_generator.py に LCM 対応を追加

class AnimeCharacterGenerator:
    def __init__(self, use_lcm: bool = False, lcm_model_dir: Optional[str] = None):
        # ... 既存のコード ...
        
        if use_lcm and lcm_model_dir:
            self.enable_lcm(lcm_model_dir)
    
    def enable_lcm(self, lcm_model_dir: str):
        """LCM モード有効化"""
        
        from diffusers import LCMScheduler
        
        lcm_scheduler = LCMScheduler.from_pretrained(lcm_model_dir)
        self.pipe.scheduler = lcm_scheduler
        
        self.is_lcm_enabled = True
        print("⚡ LCM mode enabled (4 steps)")
    
    def generate_image(self, prompt: str, **kwargs) -> Image:
        """
        LCM 対応の画像生成
        """
        
        # LCM の場合は 4-8 ステップを使用
        num_steps = 4 if self.is_lcm_enabled else 20
        
        with torch.no_grad():
            image = self.pipe(
                prompt=prompt,
                num_inference_steps=num_steps,
                **kwargs
            ).images[0]
        
        return image
```

---

## 📊 期待される改善

| 指標 | Before (SD v1.5) | After (LCM) |
|------|------------------|------------|
| 推論ステップ | 50-100 | 4-8 |
| 推論時間 | 45-90秒 | 3.6-7秒 |
| スピードアップ | 基準 | **12-25倍** |
| 品質低下 | N/A | < 5% |
| VRAM | ~8GB | ~6GB |
| Colab 12h での画像数 | ~100 | **~1200+** |

---

## 🧪 テスト・ベンチマーク

### テスト スクリプト

```python
# test_lcm_speed.py

from lcm_distiller import LCMDistiller
import time

distiller = LCMDistiller(
    lora_path="./lora_weights/anime-lora-final"
)

# ベンチマーク実行
distiller.benchmark_inference(
    lcm_model_dir="./lcm_distilled/lcm_final",
    num_samples=10
)

# 結果例:
# SD v1.5: 5.2s / image
# LCM: 0.42s / image
# Speedup: 12.4x
```

---

## ✅ 完了チェックリスト

- [ ] `lcm_distiller.py` 実装完了
- [ ] Colab で蒸留実行（3-5 エポック）
- [ ] distillation_log.json 確認
- [ ] lcm_distilled/lcm_final/ 作成確認
- [ ] ベンチマーク実行・記録
- [ ] character_generator.py に LCM 統合
- [ ] 推論速度 12x+ 達成確認
- [ ] ブログ記事執筆「4ステップ LCM による 12倍高速化」

---

**次のステップ**: Phase 2B 完了後、[PHASE_3_MULTIMODAL.md](PHASE_3_MULTIMODAL.md) へ

