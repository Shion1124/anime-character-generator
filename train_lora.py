#!/usr/bin/env python3
"""
Anime Character LoRA Training Script - Checkpoint パス実装版

このスクリプトは、Stable Diffusion v1.5 に LoRA ファインチューニングを適用します。
チェックポイント保存機能で Colab セッション切断対応。

実行例（新規学習）:
    python train_lora.py \\
        --data_dir ../training_data \\
        --output_dir ./lora_weights \\
        --epochs 20 \\
        --batch_size 2

実行例（復帰）:
    python train_lora.py \\
        --data_dir ../training_data \\
        --output_dir ./lora_weights \\
        --resume_from ./lora_weights/checkpoint-epoch-15 \\
        --epochs 20

依存パッケージ:
    pip install -q diffusers transformers pillow torch tqdm safetensors peft accelerate xformers

参考文献:
    - Ho et al. (2020): Denoising Diffusion Probabilistic Models (DDPM)
    - Rombach et al. (2022): Latent Diffusion Models
    - Hu et al. (2021): LoRA - Low-Rank Adaptation
    - Luo et al. (2023): Latent Consistency Models (LCM)

学習時間見積もり（Colab T4）:
    20 epoch × 300 images → 約 10-12 時間（分割実行推奨）
    チェックポイント: 毎 5 epoch（セッション切断対応）
"""

import os
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import time


class AnimeDataset(Dataset):
    """アニメ画像データセット"""
    
    def __init__(self, data_dir: str, resolution: int = 512):
        self.data_dir = Path(data_dir)
        self.image_paths = []
        
        # サブディレクトリから画像を収集
        for style_dir in self.data_dir.glob("*"):
            if style_dir.is_dir():
                self.image_paths.extend(list(style_dir.glob("*.png")))
                self.image_paths.extend(list(style_dir.glob("*.jpg")))
        
        if not self.image_paths:
            raise ValueError(f"❌ {data_dir} に画像が見つかりません")
        
        self.resolution = resolution
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(resolution),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        print(f"📊 データセット: {len(self.image_paths)} 画像")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
            return self.transform(image)
        except Exception as e:
            print(f"⚠️  画像読み込みエラー: {self.image_paths[idx]} - {e}")
            # エラー時はランダムな画像を返す
            return self[torch.randint(0, len(self), (1,)).item()]


class LoRATrainer:
    """
    LoRA トレーナー - チェックポイント機能付き
    
    Colab での分割実行対応：
    - 毎 5 epoch ごとにチェックポイント保存
    - --resume_from で中断復帰
    """
    
    def __init__(
        self,
        model_name: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda",
        lora_rank: int = 32,
        lora_alpha: float = 32.0,
    ):
        """
        Args:
            model_name: Hugging Face Hub のモデル名
            device: 実行デバイス
            lora_rank: LoRA ランク
            lora_alpha: LoRA アルファ値
        """
        self.model_name = model_name
        self.device = device
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.pipe = None
        
        print(f"📦 Model: {model_name}")
        print(f"💾 Device: {device}")
        print(f"🎯 LoRA Config: rank={lora_rank}, alpha={lora_alpha}")
    
    def setup_model(self):
        """Stable Diffusion + LoRA をセットアップ"""
        try:
            from diffusers import StableDiffusionPipeline
            from peft import LoraConfig, get_peft_model
        except ImportError:
            print("❌ 必須パッケージが見つかりません:")
            print("   pip install -q diffusers peft")
            raise
        
        print("\n🚀 モデル読み込み中...")
        
        # パイプライン初期化
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            safety_checker=None,
            variant="fp16"
        ).to(self.device)
        
        # VAE と Text Encoder は凍結
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)
        
        # LoRA 設定
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=["to_k", "to_v", "to_q", "to_out"],
            lora_dropout=0.1,
            bias="none"
        )
        
        # UNet に LoRA 適用
        self.pipe.unet = get_peft_model(self.pipe.unet, lora_config)
        
        # パラメータ統計
        total_params = sum(p.numel() for p in self.pipe.unet.parameters())
        trainable_params = sum(p.numel() for p in self.pipe.unet.parameters() if p.requires_grad)
        
        print(f"📊 Total UNet params: {total_params:,}")
        print(f"🎯 Trainable (LoRA) params: {trainable_params:,}")
        print(f"📉 Compression ratio: {trainable_params/total_params:.4%}")
        
        # メモリ効率化
        try:
            import xformers
            self.pipe.enable_xformers_memory_efficient_attention()
            print("✅ xFormers メモリ効率化 有効")
        except ImportError:
            print("⚠️  xFormers 非インストール（推奨: pip install xformers）")
        
        self.pipe.unet.enable_gradient_checkpointing()
        print("✅ Gradient checkpointing 有効")
        
        return self.pipe
    
    def train(
        self,
        data_dir: str,
        output_dir: str = "./lora_weights",
        epochs: int = 20,
        batch_size: int = 2,
        learning_rate: float = 1e-4,
        gradient_accumulation_steps: int = 1,
        resume_from: Optional[str] = None,
    ):
        """
        LoRA トレーニング実行
        
        Args:
            data_dir: トレーニングデータディレクトリ（training_data など）
            output_dir: LoRA ウェイト出力ディレクトリ
            epochs: トレーニングエポック数（推奨: 20）
            batch_size: バッチサイズ（T4推奨: 2）
            learning_rate: 学習率
            gradient_accumulation_steps: 勾配累積ステップ
            resume_from: チェックポイントから復帰するパス
        """
        
        # モデルセットアップ
        self.setup_model()
        
        # データセット準備
        print("\n📂 データセット準備中...")
        dataset = AnimeDataset(data_dir)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Colab対応
        )
        
        # 出力ディレクトリ作成
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # オプティマイザー
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.pipe.unet.parameters()),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # スケジューラー
        num_training_steps = len(dataloader) * epochs
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, num_training_steps
        )
        
        # ノイズスケジューラー
        from diffusers import DDPMScheduler
        noise_scheduler = DDPMScheduler.from_pretrained(
            self.model_name,
            subfolder="scheduler"
        )
        
        # 復帰処理
        start_epoch = 0
        if resume_from:
            print(f"\n🔄 チェックポイントから復帰: {resume_from}")
            checkpoint_path = Path(resume_from)
            
            if checkpoint_path.exists():
                self.pipe.unet.load_adapter(str(checkpoint_path), adapter_name="default")
                
                # メタデータから開始エポック取得
                metadata_path = checkpoint_path / "training_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                        start_epoch = metadata.get("epoch", 0)
                        print(f"✅ Epoch {start_epoch} から開始")
        
        # トレーニングループ
        self.pipe.unet.train()
        self.pipe.vae.eval()
        self.pipe.text_encoder.eval()
        
        print(f"\n" + "="*60)
        print(f"🚀 LoRA トレーニング開始")
        print(f"="*60)
        print(f"📊 Dataset: {len(dataset)} images")
        print(f"⏱️  Estimated time: 約 {epochs * 30 // 60} 時間 (Colab T4)")
        print(f"💾 Checkpoint: 毎 5 epoch ごとに保存")
        print(f"="*60 + "\n")
        
        training_log = {
            "model": self.model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "losses": []
        }
        
        total_steps = 0
        
        for epoch in range(start_epoch, epochs):
            epoch_loss = 0.0
            epoch_start_time = time.time()
            
            pbar = tqdm(
                dataloader,
                desc=f"Epoch {epoch+1}/{epochs}",
                disable=False
            )
            
            for batch_idx, pixel_values in enumerate(pbar):
                pixel_values = pixel_values.to(self.device, dtype=torch.float16)
                
                # VAE で潜在空間にエンコード
                with torch.no_grad():
                    latents = self.pipe.vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * 0.18215  # スケーリング
                
                # ノイズとタイムステップをサンプリング
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=self.device
                )
                
                # ノイズ追加
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # ダミープロンプトエンコード
                with torch.no_grad():
                    encoder_hidden_states = self.pipe.text_encoder(
                        torch.zeros(
                            latents.shape[0], 77,
                            dtype=torch.long,
                            device=self.device
                        )
                    )[0]
                
                # UNet 予測
                model_pred = self.pipe.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states
                ).sample
                
                # MSE 損失
                loss = torch.nn.functional.mse_loss(model_pred, noise)
                
                # バックプロップ
                loss.backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                lr_scheduler.step()
                
                epoch_loss += loss.item()
                total_steps += 1
                pbar.set_postfix({
                    "loss": f"{loss.item():.6f}",
                    "step": total_steps
                })
            
            avg_loss = epoch_loss / len(dataloader)
            epoch_time = time.time() - epoch_start_time
            
            print(f"  📊 Loss: {avg_loss:.6f} | ⏱️  {epoch_time:.1f}秒")
            training_log["losses"].append({
                "epoch": epoch + 1,
                "loss": avg_loss,
                "time_seconds": epoch_time
            })
            
            # チェックポイント保存（毎 5 epoch）
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                checkpoint_dir = output_path / f"checkpoint-epoch-{epoch+1}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                print(f"\n💾 チェックポイント保存: {checkpoint_dir}")
                self.pipe.unet.save_pretrained(str(checkpoint_dir))
                
                # メタデータ保存
                metadata = {
                    "epoch": epoch + 1,
                    "total_steps": total_steps,
                    "loss": avg_loss,
                    "learning_rate": learning_rate,
                }
                with open(checkpoint_dir / "training_metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)
                
                print(f"✅ チェックポイント完了\n")
        
        # 最終モデル保存
        final_dir = output_path / "anime-lora-final"
        final_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n✅ 最終モデル保存: {final_dir}")
        self.pipe.unet.save_pretrained(str(final_dir))
        
        # トレーニングログ保存
        with open(output_path / "training_log.json", "w") as f:
            json.dump(training_log, f, indent=2)
        
        print(f"\n" + "="*60)
        print(f"✅ LoRA トレーニング完了")
        print(f"="*60)
        print(f"📁 出力: {output_dir}")
        print(f"📊 総エポック: {epochs}")
        print(f"⏱️  総学習時間: {sum(log['time_seconds'] for log in training_log['losses'])/3600:.1f} 時間")
        print(f"📉 最終損失: {training_log['losses'][-1]['loss']:.6f}")
        print(f"\n🚀 次のステップ:")
        print(f"   HuggingFace へアップロード:")
        print(f"   python upload_to_huggingface.py \\")
        print(f"       --model-path {final_dir} \\")
        print(f"       --repo-name anime-character-lora")
        print(f"="*60)
        
        return self.pipe


def main():
    parser = argparse.ArgumentParser(
        description="Anime Character LoRA Training - チェックポイント対応版",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 新規学習（20 epoch）
  python train_lora.py \\
    --data_dir ../training_data \\
    --output_dir ./lora_weights \\
    --epochs 20 \\
    --batch_size 2

  # チェックポイントから復帰（15 epoch が終わっている場合）
  python train_lora.py \\
    --data_dir ../training_data \\
    --output_dir ./lora_weights \\
    --resume_from ./lora_weights/checkpoint-epoch-15 \\
    --epochs 20

学習時間見積もり（Colab T4）:
  1 epoch ≈ 30-40 分
  20 epoch ≈ 10-12 時間（分割実行推奨）
  チェックポイント保存で セッション切断後に復帰可能
        """
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../training_data",
        help="トレーニングデータディレクトリ（デフォルト: ../training_data）"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./lora_weights",
        help="LoRA ウェイト出力ディレクトリ（デフォルト: ./lora_weights）"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="トレーニングエポック数（推奨: 20, デフォルト: 20）"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="バッチサイズ（Colab T4 推奨: 2, デフォルト: 2）"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="学習率（デフォルト: 1e-4）"
    )
    
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=32,
        help="LoRA ランク（推奨: 32, デフォルト: 32）"
    )
    
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="チェックポイントから復帰（例: ./lora_weights/checkpoint-epoch-15）"
    )
    
    args = parser.parse_args()
    
    # トレーナー初期化
    trainer = LoRATrainer(
        lora_rank=args.lora_rank,
        lora_alpha=float(args.lora_rank)  # alpha = rank
    )
    
    # トレーニング実行
    trainer.train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()

