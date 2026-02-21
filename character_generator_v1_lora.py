#!/usr/bin/env python3
"""
anime-character-generator v1.5 (LoRA Edition)
Stable Diffusion v1.5 + LoRA Fine-tuning

【バージョン情報】
Version: 1.5
Date: 2026-02-17 ブログ執筆時
Status: LoRA 実装版（ブログの未着手状態から実装）

【実装内容】
- Stable Diffusion v1.5 のベースモデル
- PEFT ライブラリによる LoRA (Low-Rank Adaptation)
- Google Colab T4 GPU での実行想定
- Float16 精度、Attention Slicing による最適化

【既知の課題】 ⚠️
このバージョンは試行錯誤の結果版です。以下の課題があります：

1. Character-level noise への脆弱性
   - Gao et al. (2306.13103) が指摘する taipo/glyph 攻撃に対応していない
   - 単一レイヤーのプロンプト設計のため
   解決方法: v2.0 Phase 1 で Gemini LLM による多層冗長プロンプト設計

2. 推論速度が遅い
   - 20 ステップで 3.8秒/画像 (T4 GPU)
   - Latent Consistency Models (LCM) による 12x 高速化機会を未活用
   解決方法: v2.0 Phase 2B で LCM 蒸留を実装

3. マルチモーダル入力未対応
   - テキスト入力のみ
   - Image-to-Image, ControlNet, スケッチ入力など未実装
   解決方法: v2.0 Phase 3 で完全なマルチモーダル対応

4. 本番環境対応なし
   - 研究スクリプト形式
   - REST API, Web UI, クラウドデプロイメント未実装
   解決方法: v2.0 Phase 4 で Streamlit UI + FastAPI + クラウドデプロイ実装

これらの課題は v2.0 (Phase 1-4) で段階的に解決されます。
詳細は: IMPLEMENTATION_ROADMAP.md を参照

【論文ベース】
- Ho et al. (2020): DDPM の基礎理論
  URL: https://arxiv.org/abs/2006.11239
- Rombach et al. (2022): Stable Diffusion v1.5
  URL: https://arxiv.org/abs/2112.10752
- Hu et al. (2021): LoRA - Low-Rank Adaptation
  URL: https://arxiv.org/abs/2106.09685
- Gao et al. (2306.13103): Text-to-Image Robustness
  URL: https://arxiv.org/abs/2306.13103
"""

from diffusers import StableDiffusionPipeline
import torch
from pathlib import Path
from typing import Dict, Optional
from PIL import Image
import os
import re
import argparse

class LoRACharacterGenerator:
    """アニメキャラクター生成パイプライン（LoRA推論版）"""
    
    def __init__(
        self,
        device: str = "auto",
        model_id: str = "runwayml/stable-diffusion-v1-5",
        lora_path: Optional[str] = None,
        lora_rank: int = 32
    ):
        """
        初期化処理
        
        Args:
            device: 実行デバイス ('cuda', 'cpu', or 'auto')
            model_id: Hugging Face のモデル ID
            lora_path: LoRA ウェイトのパス（推論用）
            lora_rank: LoRA ランク（学習時と一致する必要あり）
        
        Raises:
            RuntimeError: GPU が利用不可な場合
        """
        # デバイス決定
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.lora_path = lora_path
        self.lora_rank = lora_rank
        
        print(f"📱 Device: {self.device}")
        print(f"📊 Precision: {self.dtype}")
        
        if self.device == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        # モデルロード
        print(f"\n📦 Loading {model_id}...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            safety_checker=None  # 推論高速化
        )
        self.pipe = self.pipe.to(self.device)
        self.pipe.enable_attention_slicing()  # メモリ最適化
        
        # LoRA ウェイトロード（推論用）
        if lora_path:
            self._load_lora_weights(lora_path)
        
        print("✅ Model loaded successfully")
        
        # 生成プロンプト定義
        self.base_prompt = "1girl, anime character, masterpiece, high quality"
        self.emotions = {
            "happy": "happy smile, cheerful, joyful",
            "angry": "angry expression, intense eyes",
            "sad": "sad expression, melancholic",
            "surprised": "surprised expression, wide eyes"
        }
        self.styles = {
            "with_hat": "wearing hat, stylish, fashionable",
            "with_earrings": "wearing earrings, jewelry, elegant",
            "with_makeup": "with makeup, beautiful, glamorous",
            "formal": "wearing formal dress, elegant, professional",
            "casual": "casual outfit, relaxed, friendly",
            "long_hair": "long brown hair, soft flowing hair",
            "blush": "soft blush on cheeks",
            "fireplace": "warm fireplace in background",
            "warm_lighting": "warm ambient lighting, soft orange glow",
            "cozy_room": "cozy indoor setting",
            "bokeh": "cinematic bokeh lights",
            "portrait": "upper body portrait",
            "depth_of_field": "shallow depth of field",
            "high_detail": "highly detailed",
            "soft_shading": "soft anime shading",
            "masterpiece": "masterpiece, best quality"
        }
    
    def _load_lora_weights(self, lora_path: str) -> None:
        """
        LoRA ウェイトを読み込む（推論用）
        
        Args:
            lora_path: LoRA ウェイトのパス
        """
        try:
            from peft import PeftModel
        except ImportError:
            print("❌ PEFT ライブラリが見つかりません")
            print("   pip install peft")
            raise
        
        print(f"\n🔄 Loading LoRA weights: {lora_path}")
        
        if not Path(lora_path).exists():
            raise FileNotFoundError(f"LoRA path not found: {lora_path}")
        
        # UNet に LoRA ウェイトを適用
        try:
            self.pipe.unet = PeftModel.from_pretrained(
                self.pipe.unet,
                lora_path,
                adapter_name="default"
            )
            print(f"✅ LoRA weights loaded: {lora_path}")
        except Exception as e:
            print(f"❌ LoRA loading error: {e}")
            raise
    
    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "low quality, blurry",
        num_steps: int = 20,
        guidance_scale: float = 7.0,
        seed: int = None
    ) -> Image.Image:
        """
        単一画像生成（LoRA 適用版）
        
        ⚠️  注意: v1.5 の単一レイヤープロンプト設計では、
        タイポやグリフ攻撃に対応していません。
        v2.0 は Gemini LLM による多層冗長プロンプトで対応予定。
        """
        
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        
        with torch.no_grad():
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                height=512,
                width=512
            ).images[0]
        
        return image
    
    def generate_collection(
        self,
        collection_type: str = "all",
        output_dir: str = "./outputs"
    ) -> Dict[str, str]:
        """
        複数バリエーション生成（LoRA 適用版）
        
        Args:
            collection_type: 'emotions', 'styles', 'all'
            output_dir: 出力ディレクトリ
        
        Returns:
            {name: filepath} の辞書
        """
        output_path = Path(output_dir) / f"{collection_type}_lora"
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {}
        prompts_to_generate = {}
        
        if collection_type in ["emotions", "all"]:
            prompts_to_generate.update(self.emotions)
        if collection_type in ["styles", "all"]:
            prompts_to_generate.update(self.styles)
        
        total = len(prompts_to_generate)
        
        print(f"🎨 Generating {total} images with LoRA...")
        
        for idx, (name, desc) in enumerate(prompts_to_generate.items(), 1):
            full_prompt = f"{self.base_prompt}, {desc}"
            
            print(f"[{idx}/{total}] Generating: {name}...", end="", flush=True)
            
            # メモリ清理
            torch.cuda.empty_cache()
            
            try:
                image = self.generate_image(full_prompt)
                filepath = output_path / f"character_{name}_lora.png"
                image.save(str(filepath))
                results[name] = str(filepath)
                print(" ✅")
            
            except Exception as e:
                print(f" ❌ Error: {e}")
                continue
        
        return results
    
    def _get_next_version(self, base_filename: str) -> str:
        """ファイルの次のバージョン番号を取得"""
        output_dir = "./outputs"
        existing_files = os.listdir(output_dir) if os.path.exists(output_dir) else []
        
        pattern = rf'^{re.escape(base_filename)}_v(\d+)\.png$'
        versions = []
        
        for fn in existing_files:
            match = re.match(pattern, fn)
            if match:
                versions.append(int(match.group(1)))
        
        next_version = max(versions) + 1 if versions else 1
        return f"{base_filename}_v{next_version}.png"


def main():
    parser = argparse.ArgumentParser(
        description="Anime character generator v1.5 (LoRA Edition)",
        epilog="""
使用例:

1. LoRA ウェイトなしで実行（ベースモデルのみ）:
   python character_generator_v1_lora.py --all

2. LoRA ウェイト適用で実行:
   python character_generator_v1_lora.py \\
     --lora_path ./lora_weights/anime-lora-final \\
     --all

3. 特定の感情を生成:
   python character_generator_v1_lora.py \\
     --lora_path ./lora_weights/anime-lora-final \\
     --emotion happy

【既知の課題】
- タイポ/グリフ攻撃に対応していない (v2.0 Phase 1 で解決予定)
- 推論速度が遅い (v2.0 Phase 2B で LCM 蒸留で 12 倍高速化予定)
- マルチモーダル入力未対応 (v2.0 Phase 3 で対応予定)

詳細: IMPLEMENTATION_ROADMAP.md 参照
        """
    )
    
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="LoRA ウェイトのパス（例: ./lora_weights/anime-lora-final）"
    )
    
    parser.add_argument(
        "--emotion",
        choices=["happy", "angry", "sad", "surprised"],
        help="Generate specific emotion"
    )
    
    parser.add_argument(
        "--style",
        choices=[
            "with_hat", "with_earrings", "with_makeup", "formal", "casual",
            "long_hair", "blush", "fireplace", "warm_lighting", "cozy_room",
            "bokeh", "portrait", "depth_of_field", "high_detail", "soft_shading", "masterpiece"
        ],
        help="Generate specific style"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all variations"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory (default: ./outputs)"
    )
    
    args = parser.parse_args()
    
    # ジェネレータ初期化
    generator = LoRACharacterGenerator(lora_path=args.lora_path)
    
    # 生成実行
    if args.all:
        print("\n" + "="*60)
        print("🎨 LoRA Edition - Full Collection Generation")
        print("="*60)
        results = generator.generate_collection(
            collection_type="all",
            output_dir=args.output_dir
        )
    elif args.emotion or args.style:
        # 特定の組み合わせ
        prompt_parts = [generator.base_prompt]
        if args.emotion:
            prompt_parts.append(generator.emotions[args.emotion])
        if args.style:
            prompt_parts.append(generator.styles[args.style])
        
        prompt = ", ".join(prompt_parts)
        image = generator.generate_image(prompt)
        
        emotion_part = args.emotion or "any"
        style_part = args.style or "any"
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filepath = output_path / f"character_{emotion_part}_{style_part}_lora.png"
        image.save(str(filepath))
        
        results = {f"{emotion_part}_{style_part}": str(filepath)}
    else:
        parser.print_help()
        return
    
    print(f"\n✅ Generation complete! Generated {len(results)} images")
    
    if args.lora_path:
        print(f"📌 LoRA weights: {args.lora_path}")
    else:
        print(f"⚠️  LoRA weights not loaded (using base model only)")


if __name__ == "__main__":
    main()
