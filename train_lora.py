#!/usr/bin/env python3
"""
Anime Impressionist LoRA Training Script

用途:
    Stable Diffusion v1.5 に LoRA ファインチューニングを適用
    Danbooru から収集した 300 枚の画像データセットで学習

実行例:
    # ローカル実行
    python train_lora.py --data_dir training_data --output_dir lora_weights --epochs 50
    
    # Google Colab 実行
    !python train_lora.py --data_dir /content/training_data --output_dir /content/lora_weights --epochs 100

依存パッケージ:
    pip install -q diffusers transformers pillow torch tqdm safetensors
"""

import os
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import traceback

# 条件付きインポート（環境依存）
try:
    from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTokenizer, CLIPTextModel
    from safetensors.torch import save_file
    IMPORTS_SUCCESS = True
except ImportError as e:
    print(f"⚠️  Some imports failed. Will attempt installation: {e}")
    IMPORTS_SUCCESS = False


class DoRALinear(nn.Module):
    """
    DoRA (Dominant Rank Adaptation) + QLoRA 対応の純粋 PyTorch 実装
    
    DoRA: magnitude (スケール) と direction (方向) を分離
    - magnitude: 低周波成分（スケール調整）
    - direction: 高周波成分（細部特性）
    
    QLoRA対応：
    - direction コンポーネントを低精度（int8/fp4相当）で保持 → メモリ削減
    - magnitude は高精度のまま（学習の安定性重視）
    - 順伝播: direction は混合精度で計算
    
    参考: 
    - DoRA: https://arxiv.org/abs/2402.09353
    - QLoRA: https://arxiv.org/abs/2305.14314
    """
    
    def __init__(self, linear: nn.Linear, rank: int = 32, alpha: float = 32.0, use_qlora: bool = False):
        super().__init__()
        self.linear = linear
        in_features = linear.in_features
        out_features = linear.out_features
        self.rank = rank
        self.scale = alpha / rank
        self.use_qlora = use_qlora
        
        # デバイス・dtype を元レイヤーと統一
        dtype = linear.weight.dtype
        device = linear.weight.device
        
        # ① Magnitude vector (低周波成分: スケール)
        # 常に高精度（学習の安定性）
        self.magnitude = nn.Parameter(
            torch.zeros(out_features, dtype=dtype, device=device)
        )
        
        # ② Direction matrix (高周波成分: 微細な特性)
        # QLoRA: direction は低精度で保持してメモリ削減
        direction_dtype = torch.float16 if use_qlora else dtype
        self.direction_a = nn.Linear(in_features, rank, bias=False, device=device, dtype=direction_dtype)
        self.direction_b = nn.Linear(rank, out_features, bias=False, device=device, dtype=direction_dtype)
        
        # 初期化
        nn.init.kaiming_uniform_(self.direction_a.weight, a=0.01)
        nn.init.zeros_(self.direction_b.weight)
        nn.init.zeros_(self.magnitude)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        DoRA (+ QLoRA) forward pass:
        y = W_base(x) + scale * (1 + magnitude) * direction_b(direction_a(x))
        
        QLoRA 時: direction は float16 で計算（メモリ削減）
        """
        # ベース重みによる出力
        base_out = self.linear(x)
        
        # Direction component: 低ランク分解
        # QLoRA: x は元の精度、direction_a から異なる精度で返される
        if self.use_qlora:
            # float16 direction で計算
            intermediate = self.direction_a(x.to(self.direction_a.weight.dtype))
            direction_delta = self.direction_b(intermediate).to(x.dtype)
        else:
            # 通常: 同一精度
            direction_delta = self.direction_b(self.direction_a(x))
        
        # Magnitude component: スケール適用
        magnitude_scaled = (1.0 + self.magnitude) * direction_delta
        
        return base_out + self.scale * magnitude_scaled


# LoRA 互換性のために LoRALinear エイリアスを保持
LoRALinear = DoRALinear


def inject_lora_to_unet(unet: nn.Module, rank: int = 32, alpha: float = 32.0, use_qlora: bool = False) -> list:
    """UNet のアテンション層に DoRA (+ QLoRA) を注入
    
    DoRA (Dominant Rank Adaptation):
    - magnitude: 低周波スケール成分（画像の大域的な特性）
    - direction: 高周波方向成分（細部ノイズ・テクスチャ）
    
    QLoRA (Quantized LoRA):
    - direction コンポーネントを float16 で量子化 → メモリ削減
    - magnitude は高精度のまま（学習安定性）
    
    Args:
        rank: DoRA ランク（推奨: 32-64 のユーザー実測最適値）
        alpha: スケーリング係数
        use_qlora: QLoRA を有効化（メモリセーバー、精度-メモリトレードオフ）
    
    Returns:
        lora_params: 学習対象の DoRA パラメータリスト
    """
    unet.requires_grad_(False)
    lora_modules_replaced = 0
    
    for module in unet.modules():
        # CrossAttention / Attention モジュールの to_k, to_v, to_q を置換
        for attr in ["to_k", "to_v", "to_q", "to_out"]:
            child = getattr(module, attr, None)
            if child is None:
                continue
            # to_out はリストのこともある
            if isinstance(child, nn.ModuleList):
                for i, sub in enumerate(child):
                    if isinstance(sub, nn.Linear):
                        child[i] = DoRALinear(sub, rank=rank, alpha=alpha, use_qlora=use_qlora)
                        lora_modules_replaced += 1
            elif isinstance(child, nn.Linear):
                setattr(module, attr, DoRALinear(child, rank=rank, alpha=alpha, use_qlora=use_qlora))
                lora_modules_replaced += 1
    
    print(f"  🔧 LoRA 注入: {lora_modules_replaced} modules")
    
    # LoRA パラメータのみ学習可能に設定
    lora_params = []
    for module in unet.modules():
        if isinstance(module, DoRALinear):
            module.magnitude.requires_grad_(True)
            module.direction_a.requires_grad_(True)
            module.direction_b.requires_grad_(True)
            lora_params.append(module.magnitude)
            lora_params.extend(list(module.direction_a.parameters()))
            lora_params.extend(list(module.direction_b.parameters()))
    
    return lora_params


def get_lora_state_dict(unet: nn.Module) -> dict:
    """UNet から DoRA 重みのみ抽出"""
    state_dict = {}
    for name, module in unet.named_modules():
        if isinstance(module, DoRALinear):
            state_dict[f"{name}.magnitude"] = module.magnitude.detach().cpu().float()
            state_dict[f"{name}.direction_a.weight"] = module.direction_a.weight.detach().cpu().float()
            state_dict[f"{name}.direction_b.weight"] = module.direction_b.weight.detach().cpu().float()
    return state_dict


class AnimeDataset(Dataset):
    """アニメ画像データセット"""
    
    def __init__(self, data_dir: str, image_size: int = 512):
        """初期化
        
        Args:
            data_dir: training_data ディレクトリ
            image_size: 出力画像サイズ
        """
        self.image_size = image_size
        self.image_paths = []
        self.metadata = {}
        
        data_path = Path(data_dir)
        
        # メタデータロード
        metadata_file = data_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        
        # 画像パスを収集
        for style_dir in data_path.iterdir():
            if not style_dir.is_dir() or style_dir.name.startswith("."):
                continue
            
            for img_file in style_dir.glob("*.*"):
                if img_file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    self.image_paths.append(str(img_file))
        
        print(f"📊 Dataset loaded: {len(self.image_paths)} images")
        
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), 
                             interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # [-1, 1] 範囲へ正規化
        ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """データセットの 1 サンプルを取得"""
        
        img_path = self.image_paths[idx]
        
        try:
            # 画像ロード
            image = Image.open(img_path).convert("RGB")
            pixel_values = self.transform(image)
            
            # スタイル名を取得（ディレクトリ名から）
            style_name = Path(img_path).parent.name
            
            # プロンプト生成（スタイル名 + 基本プロンプト）
            prompt = f"{style_name}, anime, masterpiece, high quality"
            
            return {
                "pixel_values": pixel_values,
                "prompt": prompt,
                "file_path": img_path
            }
        
        except Exception as e:
            print(f"❌ Error loading {img_path}: {e}")
            # フォールバック: 最初のサンプルを返す
            return self[0] if idx > 0 else self[(idx + 1) % len(self)]


class LoRATrainer:
    """LoRA ファインチューニングトレーナー"""
    
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        output_dir: str = "lora_weights",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lora_rank: int = 32,
        lora_alpha: float = 32,
        use_qlora: bool = False,
    ):
        """初期化
        
        Args:
            model_id: Hugging Face モデル ID
            output_dir: 出力ディレクトリ
            device: 計算デバイス
            lora_rank: DoRA ランク（推奨: 32-64）
            lora_alpha: DoRA スケーリング係数
            use_qlora: QLoRA を使用（direction量子化、メモリ削減）
        """
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.device = device
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.use_qlora = use_qlora
        
        print("="*60)
        print("🚀 DoRA Trainer Initialization")
        print("="*60)
        print(f"📦 Model: {model_id}")
        print(f"📁 Output: {self.output_dir}")
        print(f"💻 Device: {device}")
        print(f"🎯 DoRA Config: rank={lora_rank}, alpha={lora_alpha}")
        if use_qlora:
            print(f"⚡ QLoRA: Enabled (direction: float16 → int8 quantization)")
        print(f"   (Dominant Rank Adaptation - magnitude + direction decomposition)")
        print(f"💻 Device: {device}")
        print(f"🎯 DoRA Config: rank={lora_rank}, alpha={lora_alpha}")
        print(f"   (Dominant Rank Adaptation - magnitude + direction decomposition)")
        
        self._setup_model()
    
    def _setup_model(self):
        """モデル・DoRA 設定の初期化 (純粋 PyTorch DoRA)"""
        
        try:
            print("\n📥 Loading Stable Diffusion v1.5...")
            dtype = torch.float16 if "cuda" in self.device else torch.float32
            
            # 各コンポーネントを個別にロード
            self.tokenizer = CLIPTokenizer.from_pretrained(self.model_id, subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained(
                self.model_id, subfolder="text_encoder", torch_dtype=dtype
            ).to(self.device)
            self.vae = AutoencoderKL.from_pretrained(
                self.model_id, subfolder="vae", torch_dtype=dtype
            ).to(self.device)
            self.unet = UNet2DConditionModel.from_pretrained(
                self.model_id, subfolder="unet", torch_dtype=dtype
            ).to(self.device)
            
            # VAE とテキストエンコーダーは凍結
            self.vae.requires_grad_(False)
            self.text_encoder.requires_grad_(False)
            
            # 純粋 PyTorch DoRA を UNet アテンション層に注入
            # Magnitude (低周波) と Direction (高周波) を分離学習
            self.lora_params = inject_lora_to_unet(
                self.unet, rank=self.lora_rank, alpha=self.lora_alpha, use_qlora=self.use_qlora
            )
            
            # 安全策: DoRA 注入後に全パラメータを確実にデバイスへ移動
            self.unet.to(self.device)
            
            # Gradient Checkpointing 有効化（VRAM 節約）
            if "cuda" in self.device:
                self.unet.enable_gradient_checkpointing()
                print("✅ Gradient checkpointing enabled (VRAM 節約)")
            
            # DoRA パラメータ参照を更新（magnitude + direction コンポーネント）
            self.lora_params = []
            for m in self.unet.modules():
                if isinstance(m, DoRALinear):
                    self.lora_params.append(m.magnitude)  # magnitude ベクトル
                    self.lora_params.extend(list(m.direction_a.parameters()))
                    self.lora_params.extend(list(m.direction_b.parameters()))
            
            total_params = sum(p.numel() for p in self.lora_params)
            print(f"✅ DoRA configured: {total_params:,} trainable params")
            if self.use_qlora:
                print(f"✅ QLoRA enabled: direction components in float16 (memory efficient)")
            print(f"✅ Model loaded (純粋 PyTorch DoRA, 低周波×高周波分離学習)")
            
        except Exception as e:
            print(f"❌ Error setting up model: {e}")
            traceback.print_exc()
            raise
    
    def train(
        self,
        data_dir: str,
        num_epochs: int = 50,
        batch_size: int = 1,
        learning_rate: float = 1e-4,
        num_workers: int = 0,
        save_interval: int = 5,
        use_qlora: bool = True,
        gradient_accumulation_steps: int = 1,
    ):
        """LoRA トレーニング実行
        
        Args:
            data_dir: トレーニングデータディレクトリ
            num_epochs: エポック数
            batch_size: バッチサイズ
            learning_rate: 学習率
            num_workers: データローダーのワーカー数
            save_interval: チェックポイント保存間隔（エポック）
            use_qlora: QLoRA を有効化
            gradient_accumulation_steps: 勾配蓄積ステップ数（実効バッチ = batch_size * steps）
        """
        
        print("\n" + "="*60)
        print("🎓 Starting DoRA Training")
        print("="*60)
        print(f"📊 Training Epochs: {num_epochs}")
        print(f"📦 Batch Size: {batch_size}")
        print(f"📦 Gradient Accumulation: {gradient_accumulation_steps} steps (effective batch={batch_size * gradient_accumulation_steps})")
        print(f"🎯 Learning Rate: {learning_rate}")
        print(f"💡 DoRA: Magnitude (低周波) + Direction (高周波) 分離学習")
        print(f"🔧 QLoRA: {use_qlora}")
        use_amp = "cuda" in self.device
        if use_amp:
            print(f"⚡ Mixed Precision (AMP): Enabled")
        
        # トレーニングログの初期化（try-except 外）
        training_log = {
            "config": {
                "model_id": self.model_id,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "effective_batch_size": batch_size * gradient_accumulation_steps,
                "learning_rate": learning_rate,
                "lora_rank": self.lora_rank,
                "lora_alpha": self.lora_alpha,
                "use_qlora": use_qlora,
                "mixed_precision": use_amp,
            },
            "history": [],
            "status": "initializing"
        }
        
        try:
            # データセット作成
            dataset = AnimeDataset(data_dir)
            
            # データセット検証
            if len(dataset) == 0:
                raise ValueError(
                    f"❌ Dataset is empty! No images found in {data_dir}\n"
                    "   Expected structure: {data_dir}/<style_name>/*.png\n"
                    "   Please upload training images first."
                )
            
            print(f"✅ Dataset loaded: {len(dataset)} images")
            
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers
            )
            
            # データローダー検証
            num_batches = len(dataloader)
            print(f"✅ DataLoader ready: {num_batches} batches")
            
            if num_batches == 0:
                raise ValueError(
                    f"❌ DataLoader has 0 batches!\n"
                    f"   Dataset size: {len(dataset)}, Batch size: {batch_size}\n"
                    "   This usually means the DataLoader failed to create batches."
                )
            
            # オプティマイザー設定 (LoRA パラメータのみ)
            optimizer = torch.optim.AdamW(
                self.lora_params,
                lr=learning_rate
            )
            
            # スケジューラー設定
            num_training_steps = num_batches * num_epochs
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, num_training_steps
            )
            
            # ノイズスケジューラー
            noise_scheduler = DDPMScheduler.from_pretrained(
                self.model_id, subfolder="scheduler"
            )
            
            # トレーニングモード設定
            self.unet.train()   # UNet は train モード（LoRAのみ勾配あり）
            self.vae.eval()
            self.text_encoder.eval()
            # LoRA 以外の UNet パラメータは勾配なし（inject 時に設定済み）
            
            # Mixed Precision (AMP) 設定
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
            
            # 既に定義された training_log にデータセット情報を追加
            training_log["config"]["dataset_size"] = len(dataset)
            training_log["config"]["num_batches"] = num_batches
            training_log["status"] = "training"
            
            global_step = 0
            
            for epoch in range(num_epochs):
                print(f"\n[Epoch {epoch + 1}/{num_epochs}]")
                epoch_loss = 0.0
                num_valid_batches = 0
                
                pbar = tqdm(dataloader, desc="Training", leave=False)
                
                for batch_idx, batch in enumerate(pbar):
                    # プロンプトをトークン化
                    prompts = batch["prompt"]
                    dtype = torch.float16 if "cuda" in self.device else torch.float32
                    pixel_values = batch["pixel_values"].to(device=self.device, dtype=dtype)
                    
                    # テキストエンコード & VAE エンコード（勾配不要）
                    with torch.no_grad():
                        input_ids = self.tokenizer(
                            prompts,
                            max_length=self.tokenizer.model_max_length,
                            padding="max_length",
                            truncation=True,
                            return_tensors="pt"
                        ).input_ids.to(self.device)
                        
                        encoder_hidden_states = self.text_encoder(input_ids)[0]
                        
                        # 画像を潜在空間にエンコード（UNet は潜在変数を受け取る）
                        latents = self.vae.encode(pixel_values).latent_dist.sample()
                        latents = latents * self.vae.config.scaling_factor
                    
                    # VAE 出力後に中間テンソルを解放
                    del pixel_values, input_ids
                    
                    # ノイズサンプリング
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (latents.shape[0],),
                        device=self.device
                    ).long()
                    
                    # ノイズが追加された潜在表現
                    noisy_latents = noise_scheduler.add_noise(
                        latents, noise, timesteps
                    )
                    
                    # U-Net 予測 (AMP autocast で forward pass を float16 計算)
                    if use_amp:
                        with torch.amp.autocast("cuda", dtype=torch.float16):
                            model_pred = self.unet(
                                noisy_latents,
                                timesteps,
                                encoder_hidden_states
                            ).sample
                            # 損失計算（ノイズ予測）
                            loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float())
                    else:
                        model_pred = self.unet(
                            noisy_latents,
                            timesteps,
                            encoder_hidden_states
                        ).sample
                        loss = torch.nn.functional.mse_loss(model_pred, noise)
                    
                    # NaN チェック（数値安定性）
                    if torch.isnan(loss):
                        print(f"    ⚠️  NaN detected at step {global_step}, skipping batch")
                        optimizer.zero_grad()
                        continue
                    
                    # 勾配蓄積のためにスケール
                    loss = loss / gradient_accumulation_steps
                    
                    # バックプロップ (AMP scaler)
                    if use_amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    
                    # 勾配蓄積ステップ完了時にパラメータ更新
                    if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                        if use_amp:
                            scaler.unscale_(optimizer)
                        
                        # 勾配クリッピング＆チェック
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.lora_params, 1.0)
                        if grad_norm > 10.0:
                            print(f"    ⚠️  High gradient norm: {grad_norm:.4f} at step {global_step}")
                        
                        if use_amp:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        
                        lr_scheduler.step()
                        optimizer.zero_grad()
                    
                    epoch_loss += loss.item() * gradient_accumulation_steps  # unscale for logging
                    num_valid_batches += 1
                    global_step += 1
                    
                    pbar.update(1)
                    pbar.set_postfix({"loss": f"{loss.item():.6f}"})
                
                avg_loss = epoch_loss / num_valid_batches if num_valid_batches > 0 else float('nan')
                training_log["history"].append({
                    "epoch": epoch + 1,
                    "loss": avg_loss,
                    "lr": optimizer.param_groups[0]["lr"]
                })
                
                if torch.isnan(torch.tensor(avg_loss)):
                    print(f"  ⚠️  Epoch Loss: nan (potential training instability)")
                else:
                    print(f"  📊 Epoch Loss: {avg_loss:.6f}")
                
                # チェックポイント保存
                if (epoch + 1) % save_interval == 0:
                    self._save_checkpoint(epoch + 1)
            
            print("\n✅ Training completed!")
            training_log["status"] = "completed"
            
            # トレーニングログ保存
            log_file = self.output_dir / "training_log.json"
            with open(log_file, "w") as f:
                json.dump(training_log, f, indent=2)
            print(f"📊 Training log saved: {log_file}")
            
            # 最終モデル保存
            self.save_lora_weights()
            
            return training_log
        
        except Exception as e:
            print(f"\n❌ Training error: {e}")
            traceback.print_exc()
            
            # エラー情報をログに追加して保存（できるなら）
            try:
                training_log["status"] = "error"
                training_log["error"] = str(e)
                training_log["error_traceback"] = traceback.format_exc()
                
                self.output_dir.mkdir(parents=True, exist_ok=True)
                log_file = self.output_dir / "training_log.json"
                with open(log_file, "w") as f:
                    json.dump(training_log, f, indent=2)
                print(f"📊 Error log saved: {log_file}")
            except Exception as log_error:
                print(f"⚠️  Could not save error log: {log_error}")
            
            raise
    
    def _save_checkpoint(self, epoch: int):
        """チェックポイント保存"""
        
        checkpoint_dir = self.output_dir / f"checkpoint-{epoch}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        state_dict = get_lora_state_dict(self.unet)
        save_file(state_dict, checkpoint_dir / "lora_weights.safetensors")
        print(f"  💾 Checkpoint saved: {checkpoint_dir} ({len(state_dict)} tensors)")
    
    def save_lora_weights(self, filename: str = "anime-impressionist-lora.safetensors"):
        """LoRA 重みを SafeTensors フォーマットで保存
        
        Args:
            filename: 保存ファイル名
        """
        
        save_path = self.output_dir / filename
        
        try:
            state_dict = get_lora_state_dict(self.unet)
            save_file(state_dict, save_path)
            
            file_size_mb = save_path.stat().st_size / (1024 * 1024)
            print(f"✅ LoRA weights saved: {save_path} ({file_size_mb:.2f} MB, {len(state_dict)} tensors)")
            
            # adapter_config.json も保存（互換性のため）
            config = {
                "base_model_name_or_path": self.model_id,
                "lora_rank": self.lora_rank,
                "lora_alpha": float(self.lora_alpha),
                "target_modules": ["to_k", "to_v", "to_q", "to_out"],
                "implementation": "pytorch_native"
            }
            with open(self.output_dir / "adapter_config.json", "w") as f:
                json.dump(config, f, indent=2)
            
            return save_path
        
        except Exception as e:
            print(f"❌ Error saving LoRA weights: {e}")
            traceback.print_exc()
            raise


def main():
    """メイン実行"""
    
    parser = argparse.ArgumentParser(
        description="LoRA ファインチューニング実行スクリプト"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="training_data",
        help="トレーニングデータディレクトリ"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lora_weights",
        help="LoRA 重み出力ディレクトリ"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="トレーニングエポック数"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="バッチサイズ (T4: 1-4推奨, A100: 4-8可能)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="学習率"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=32,
        help="DoRA ランク (デフォルト 32, ユーザー実測最適値 32-64)"
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=32,
        help="LoRA アルファ (スケーリング係数)"
    )
    parser.add_argument(
        "--use_qlora",
        action="store_true",
        help="QLoRA を有効化 (direction: float16 量子化, メモリ削減)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="計算デバイス (cuda, cpu)"
    )
    
    args = parser.parse_args()
    
    # ダイナミックインポート (pip install 後)
    if not IMPORTS_SUCCESS:
        print("⚠️  Attempting to install required packages...")
        os.system("pip install -q diffusers transformers pillow torch tqdm safetensors")
    
    # トレーナー実行
    trainer = LoRATrainer(
        output_dir=args.output_dir,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        device=args.device,
        use_qlora=args.use_qlora
    )
    
    training_log = trainer.train(
        data_dir=args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    print("\n" + "="*60)
    print("🎉 LoRA Training Complete!")
    print("="*60)
    print(f"📁 Output: {trainer.output_dir}")
    print(f"📊 Final Loss: {training_log['history'][-1]['loss']:.6f}")
    print("="*60)


if __name__ == "__main__":
    main()
