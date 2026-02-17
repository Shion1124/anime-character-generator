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
from PIL import Image
import matplotlib.pyplot as plt


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
            "with_hat": "wearing hat",
            "with_earrings": "wearing earrings",
            "formal": "formal dress, elegant",
            "casual": "casual outfit",
            "with_makeup": "with makeup, beautiful",
            "glasses": "wearing glasses"
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
        
        # 感情結果表示
        self._display_results(emotion_images, "emotion_results.png", rows=2, cols=2)
        
        # スタイル結果表示
        self._display_results(style_images, "style_results.png", rows=2, cols=3)
        
        print("\n" + "="*60)
        print("✅ GENERATION COMPLETE!")
        print("="*60)
        print(f"\n📁 Generated {len(emotion_images) + len(style_images)} images")
        print(f"📁 Output directory: ./outputs/")
    
    def _display_results(self, images: dict, output_file: str, rows: int, cols: int):
        """結果画像を表示・保存"""
        fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*5))
        axes = axes.flatten() if rows * cols > 1 else [axes]
        
        for idx, (name, img) in enumerate(list(images.items())[:rows*cols]):
            axes[idx].imshow(img)
            axes[idx].set_title(name.upper(), fontsize=12, fontweight='bold')
            axes[idx].axis('off')
        
        # 余った軸を非表示
        for idx in range(len(images), rows*cols):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="アニメキャラクター自動生成")
    parser.add_argument("--emotion", choices=["happy", "angry", "sad", "surprised"],
                       help="感情を指定")
    parser.add_argument("--style", choices=["with_hat", "with_earrings", "formal", "casual", "with_makeup", "glasses"],
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
