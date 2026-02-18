#!/usr/bin/env python3
"""
anime-character-generator
Stable Diffusion + PyTorch を活用した、アニメキャラクター自動生成

Usage:
    python character_generator.py --emotion happy --style casual
    python character_generator.py --all
"""

import argparse
import os
import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont
import re


class AnimeCharacterGenerator:
    def __init__(self, device: str = "auto"):
        """
        初期化
        
        Args:
            device: 実行デバイス ('cuda', 'cpu', or 'auto')
        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        print(f"📦 Device: {self.device} | Dtype: {self.dtype}")
        print(f"✓ GPU Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        
        # モデルロード
        print("\n📦 Loading Stable Diffusion v1.5...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=self.dtype,
            safety_checker=None
        )
        self.pipe = self.pipe.to(self.device)
        self.pipe.enable_attention_slicing()
        print("✅ Model ready!")
        
        # ベースプロンプト
        self.base_prompt = "1girl, anime character, masterpiece, high quality"
        
        # 感情定義
        self.emotions = {
            "happy": "happy smile, cheerful, joyful",
            "angry": "angry expression, intense eyes",
            "sad": "sad expression, melancholic",
            "surprised": "surprised expression, wide eyes"
        }
        
        # スタイル定義
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
        output_path: str = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.0,
        height: int = 512,
        width: int = 512,
        seed: int = None
    ) -> Image.Image:
        """
        単一画像生成
        
        Args:
            prompt: プロンプト
            output_path: 保存先パス
            num_inference_steps: 推論ステップ数
            guidance_scale: ガイダンススケール
            height: 画像高さ
            width: 画像幅
            seed: 乱数シード
            
        Returns:
            PIL Image
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        with torch.no_grad():
            image = self.pipe(
                prompt=prompt,
                negative_prompt="low quality, blurry",
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width
            ).images[0]
        
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            image.save(output_path)
            print(f"  ✅ Saved: {output_path}")
        
        return image
    
    def generate_emotions(self, output_dir: str = "./outputs/emotions") -> dict:
        """
        全感情バリエーション生成
        
        Returns:
            {emotion_name: PIL.Image} の辞書
        """
        print("\n🎭 GENERATING EMOTIONS...\n")
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        for emotion_name, emotion_desc in self.emotions.items():
            prompt = f"{self.base_prompt}, {emotion_desc}"
            print(f"  [{emotion_name.upper()}] Generating...", end="", flush=True)
            
            filepath = os.path.join(output_dir, f"character_{emotion_name}.png")
            image = self.generate_image(prompt, filepath)
            results[emotion_name] = image
        
        print(f"\n✅ Emotions generation complete!")
        return results
    
    def generate_styles(self, output_dir: str = "./outputs/styles") -> dict:
        """
        全スタイルバリエーション生成
        
        Returns:
            {style_name: PIL.Image} の辞書
        """
        print("\n👗 GENERATING STYLES...\n")
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        for style_name, style_desc in self.styles.items():
            prompt = f"{self.base_prompt}, {style_desc}"
            print(f"  [{style_name.upper()}] Generating...", end="", flush=True)
            
            filepath = os.path.join(output_dir, f"character_{style_name}.png")
            image = self.generate_image(prompt, filepath)
            results[style_name] = image
        
        print(f"\n✅ Styles generation complete!")
        return results
    
    def generate_all(self):
        """全パターン生成 + 結果表示"""
        emotion_images = self.generate_emotions()
        style_images = self.generate_styles()
        
        # グリッド形式で合成
        print("\n📊 Creating composite grid images...")
        self._create_grid_composite(emotion_images, "emotion_results", rows=2, cols=2)
        self._create_grid_composite(style_images, "style_results", rows=2, cols=4)
        
        print("\n" + "="*60)
        print("✅ GENERATION COMPLETE!")
        print("="*60)
        print(f"\n📁 Generated {len(emotion_images) + len(style_images)} images")
        print(f"📁 Output directory: ./outputs/")
    
    def _get_next_version(self, base_filename: str) -> str:
        """
        ファイルの次のバージョン番号を取得
        例: style_results_v1.png → style_results_v2.png
        """
        output_dir = "./outputs"
        existing_files = []
        
        if os.path.exists(output_dir):
            existing_files = os.listdir(output_dir)
        
        # ベースファイル名に合致するファイルを検索
        pattern = rf'^{re.escape(base_filename)}_v(\d+)\.png$'
        versions = []
        
        for fn in existing_files:
            match = re.match(pattern, fn)
            if match:
                versions.append(int(match.group(1)))
        
        # 次のバージョンは最大値+1、ない場合は1
        next_version = max(versions) + 1 if versions else 1
        return f"{base_filename}_v{next_version}.png"
    
    def _create_grid_composite(
        self, 
        images: dict, 
        base_filename: str,
        rows: int = 2, 
        cols: int = 2,
        img_size: int = 512,
        gap: int = 10
    ):
        """
        複数画像をグリッドレイアウトで合成
        
        Args:
            images: {name: PIL.Image} の辞書
            base_filename: 出力ファイル名（バージョン番号なし）
            rows: グリッド行数
            cols: グリッド列数
            img_size: 各画像のサイズ
            gap: 画像間のギャップ
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
        print(f"  ✅ Saved: {output_filename}")
    
    def _display_results(self, images: dict, output_file: str, rows: int, cols: int):
        """結果画像を表示・保存（非推奨：互換性で残す）"""
        # このメソッドは _create_grid_composite に置き換わった
        pass


def main():
    parser = argparse.ArgumentParser(description="アニメキャラクター自動生成")
    parser.add_argument("--emotion", choices=["happy", "angry", "sad", "surprised"],
                       help="感情を指定")
    parser.add_argument("--style", 
                       choices=["with_hat", "with_earrings", "with_makeup", "formal", "casual",
                               "long_hair", "blush", "fireplace", "warm_lighting", "cozy_room",
                               "bokeh", "portrait", "depth_of_field", "high_detail", 
                               "soft_shading", "masterpiece"],
                       help="スタイルを指定")
    parser.add_argument("--all", action="store_true", help="全パターン生成")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="auto",
                       help="実行デバイス")
    
    args = parser.parse_args()
    
    # ジェネレータ初期化
    generator = AnimeCharacterGenerator(device=args.device)
    
    if args.all:
        generator.generate_all()
    elif args.emotion and args.style:
        # 特定の感情+スタイルで生成
        emotion_desc = generator.emotions[args.emotion]
        style_desc = generator.styles[args.style]
        prompt = f"{generator.base_prompt}, {emotion_desc}, {style_desc}"
        print(f"\n🎨 Generating: {args.emotion} + {args.style}")
        image = generator.generate_image(prompt)
        image.show()
    elif args.emotion:
        # 感情のみで生成
        generator.generate_emotions()
    elif args.style:
        # スタイルのみで生成
        generator.generate_styles()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
