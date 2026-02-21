#!/usr/bin/env python3
"""
anime-character-generator v1.0
PyTorch + Diffusers を用いたアニメキャラクター生成ツール

【バージョン情報】
Version: 1.0
Date: 2026-02-17
Status: ブログで完全説明される基本実装版

実装上の工夫：
1. GPU メモリ管理の最適化
2. バッチ処理による効率化
3. エラー時の安全な処理
4. 詳細なログ出力

【論文ベース】
- Ho et al. (2020): DDPM の基礎理論
  URL: https://arxiv.org/abs/2006.11239
- Rombach et al. (2022): Stable Diffusion v1.5
  URL: https://arxiv.org/abs/2112.10752
"""

from diffusers import StableDiffusionPipeline
import torch
from pathlib import Path
from datetime import datetime
import argparse
import json
import os
import re
from typing import Dict, Tuple
from PIL import Image

class AnimeCharacterGenerator:
    """アニメキャラクター生成パイプライン"""
    
    def __init__(self, device: str = "auto", model_id: str = "runwayml/stable-diffusion-v1-5"):
        """
        初期化処理
        
        Args:
            device: 実行デバイス ('cuda', 'cpu', or 'auto')
            model_id: Hugging Face のモデル ID
        
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
    
    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "low quality, blurry",
        num_steps: int = 20,
        guidance_scale: float = 7.0,
        seed: int = None
    ) -> Image.Image:
        """単一画像生成"""
        
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
        複数バリエーション生成
        
        Args:
            collection_type: 'emotions', 'styles', 'all'
            output_dir: 出力ディレクトリ
        
        Returns:
            {name: filepath} の辞書
        """
        output_path = Path(output_dir) / collection_type
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {}
        prompts_to_generate = {}
        
        if collection_type in ["emotions", "all"]:
            prompts_to_generate.update(self.emotions)
        if collection_type in ["styles", "all"]:
            prompts_to_generate.update(self.styles)
        
        total = len(prompts_to_generate)
        
        for idx, (name, desc) in enumerate(prompts_to_generate.items(), 1):
            full_prompt = f"{self.base_prompt}, {desc}"
            
            print(f"[{idx}/{total}] Generating: {name}...", end="", flush=True)
            
            # メモリ清理
            torch.cuda.empty_cache()
            
            try:
                image = self.generate_image(full_prompt)
                filepath = output_path / f"character_{name}.png"
                image.save(str(filepath))
                results[name] = str(filepath)
                print(" ✅")
            
            except Exception as e:
                print(f" ❌ Error: {e}")
                continue
        
        return results
    
    def _get_next_version(self, base_filename: str) -> str:
        """
        ファイルの次のバージョン番号を取得
        例: style_results_v1.png → style_results_v2.png
        """
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
    
    def _create_grid_composite(
        self, 
        images: Dict[str, Image.Image], 
        base_filename: str,
        rows: int = 2, 
        cols: int = 2,
        img_size: int = 512,
        gap: int = 10
    ) -> Image.Image:
        """
        複数画像をグリッドレイアウトで合成
        
        Args:
            images: {name: PIL.Image} の辞書
            base_filename: 出力ファイル名（バージョン番号なし）
            rows: グリッド行数
            cols: グリッド列数
            img_size: 各画像のサイズ
            gap: 画像間のギャップ
        
        Returns:
            合成済みの PIL Image
        """
        os.makedirs("./outputs", exist_ok=True)
        
        # 使用する画像を取得（最大 rows*cols）
        use_images = list(images.items())[:rows*cols]
        
        # キャンバスサイズ計算
        canvas_width = cols * img_size + (cols - 1) * gap + gap * 2
        canvas_height = rows * img_size + (rows - 1) * gap + gap * 2
        
        # キャンバス作成（白背景）
        canvas = Image.new('RGB', (canvas_width, canvas_height), color='white')
        
        # 各画像をペースト
        for idx, (name, img) in enumerate(use_images):
            row = idx // cols
            col = idx % cols
            
            # ペースト位置
            x = gap + col * (img_size + gap)
            y = gap + row * (img_size + gap)
            
            # 画像をリサイズしてペースト
            resized_img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
            canvas.paste(resized_img, (x, y))
        
        # 次のバージョン番号を取得して保存
        output_filename = self._get_next_version(base_filename)
        output_path = f"./outputs/{output_filename}"
        
        canvas.save(output_path, quality=95)
        print(f"✅ Saved: {output_filename}")
        
        return canvas


def main():
    parser = argparse.ArgumentParser(description="Anime character generator v1.0")
    parser.add_argument("--emotion", choices=list(AnimeCharacterGenerator().emotions.keys()),
                       help="Generate specific emotion")
    parser.add_argument("--style", choices=list(AnimeCharacterGenerator().styles.keys()),
                       help="Generate specific style")
    parser.add_argument("--all", action="store_true", help="Generate all variations")
    
    args = parser.parse_args()
    
    # ジェネレータ初期化
    generator = AnimeCharacterGenerator()
    
    # 生成実行
    if args.all:
        results = generator.generate_collection(collection_type="all")
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
        image.save(f"character_{emotion_part}_{style_part}.png")
        results = {f"{emotion_part}_{style_part}": "generated"}
    else:
        parser.print_help()
        return
    
    print(f"\n✅ Generation complete! Generated {len(results)} images")


if __name__ == "__main__":
    main()
