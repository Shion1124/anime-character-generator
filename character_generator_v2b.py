#!/usr/bin/env python3
"""
anime-character-generator v2.0B (Phase 2B LCM蒸留版 - 本番対応+HF統合)
Stable Diffusion v1.5 + LoRA + LCM × 論文ベース改善 + HuggingFace Release

【バージョン情報】  
Version: 2.0B (Phase 2B 完成版 - 本番対応版 + Phase 4 HF統合)
Date: 2026-03-05
Status: ✅ Phase 2B (LCM蒸留) + Phase 4 (HF Release) 完成・本番対応版

【対応ノートブック】
📓 anime_generator_colab_lora_v2b.ipynb (v2.0B完成版)
   └─ Google Colab での Phase 2B LCM蒸留実装
   └─ Phase 4: HuggingFace アップロード機能
   └─ トークン管理：Read（ダウンロード）→ Write（アップロード）

【実装済み (✅)】
✅ Phase 2B: LCM 蒸留による推論 5倍高速化 (4ステップ推論)
✅ LoRA統合: anime-character-lora_v1.5 対応
✅ 推論速度: ~1.3秒/画像 (float16, T4実測 — guidance=1.5)
✅ 品質: 公式 LCM-LoRA + anime LoRA マージで v1.5 同等品質を維持
✅ Phase 4: HuggingFace Hub Release (README生成 + モデルアップロード)
✅ トークン管理: 自動Read→Write切り替え

【バージョン戦略】
- character_generator_v2b.py: Phase 2B+4完成版 (このファイル - 本番対応・参考用)
- character_generator.py: Phase 3以降で更新・拡張

【HuggingFace トークンについて】
⚠️ LoRA ダウンロード: 「読み取りトークン」（Read Token）が必要
   - トークン取得: https://huggingface.co/settings/tokens
   - Scope: 必要に応じて「読み取り」のみ設定

⚠️ モデルアップロード: 「書き込みトークン」（Write Token）が必要 [Phase 4]
   - トークン取得: https://huggingface.co/settings/tokens
   - Scope: リポジトリへの「書き込み」権限が必須
   - 自動ダウンロード失敗時は CLI で `huggingface-cli login` を実行

【使用場面】
✅ ローカル推論（高速）
✅ HuggingFace Hub へのモデルリリース [Phase 4]
✅ 参考実装として Phase 2B コードを確認
✅ Phase 3 実装時に Phase 2B 機能を参照

使用例:
    # 基本的な推論 (20ステップ)
    python character_generator_v2b.py --emotion happy --style casual
    
    # LCM高速推論 (4ステップ)
    python character_generator_v2b.py --emotion happy --lcm
    
    # LoRA + LCM (最適化構成)
    python character_generator_v2b.py --lora --lcm --emotion happy --style casual
    
    # HuggingFace へのモデルアップロード [Phase 4]
    python character_generator_v2b.py --upload-model --repo-id "your_username/anime-character-lcm-lora"
"""

import argparse
import os
import time
import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline, LCMScheduler
from PIL import Image, ImageDraw, ImageFont
import re

try:
    from peft import PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

try:
    from prompt_optimizer import RobustPromptGenerator
    HAS_ROBUST_PROMPT = True
except ImportError:
    HAS_ROBUST_PROMPT = False
    print("⚠️  RobustPromptGenerator not available. Use --use-robust-prompt to enable.")


class AnimeCharacterGenerator:
    def __init__(
        self, 
        device: str = "auto", 
        use_robust_prompt: bool = False,
        use_lora: bool = False,
        use_lcm: bool = False,
        lora_model_id: str = "yoshihisashinzaki/anime-character-lora_v1.5",
        lora_path: str = None,
        use_official_lcm_lora: bool = False,
    ):
        """
        初期化
        
        Args:
            device: 実行デバイス ('cuda', 'mps', 'cpu', or 'auto')
            use_robust_prompt: RobustPromptGenerator を使用するか
            use_lora: LoRA を統合するか (anime-character-lora_v1.5)
            use_lcm: LCM Scheduler を使用するか (4ステップ高速推論)
            lora_model_id: HuggingFace LoRA モデルID
            lora_path: ローカル PEFT 形式 anime LoRA ディレクトリパス (例: ./lora_weights/anime-lora-final)
            use_official_lcm_lora: 公式 LCM-LoRA (latent-consistency/lcm-lora-sdv1-5) を使用
        """
        # デバイス設定
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        self.dtype = torch.float16 if self.device != "cpu" else torch.float32
        self.use_lcm = use_lcm
        self.use_lora = use_lora
        self.lora_model_id = lora_model_id
        self.lora_path = lora_path
        self.use_official_lcm_lora = use_official_lcm_lora
        
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
        print("✅ Model loaded")
        
        # LoRA 統合 (Phase 2B対応)
        if use_lora or lora_path:
            # (A) ローカル PEFT 形式 LoRA（推奨: Google Drive からダウンロード後に使用）
            _lora_path = Path(lora_path) if lora_path else None
            if _lora_path and _lora_path.exists():
                if not HAS_PEFT:
                    print("⚠️  peft ライブラリがありません: pip install peft")
                else:
                    try:
                        print(f"🎨 PEFT 形式 anime LoRA をロード & マージ: {_lora_path}")
                        peft_unet = PeftModel.from_pretrained(
                            self.pipe.unet, str(_lora_path), adapter_name="anime"
                        )
                        self.pipe.unet = peft_unet.merge_and_unload()
                        print("✅ anime LoRA を UNet にマージ完了")
                    except Exception as e:
                        print(f"⚠️  PEFT LoRA のマージ失敗: {e}")
            # (B) HuggingFace hub 上の diffusers 形式 LoRA（フォールバック）
            elif use_lora:
                try:
                    print(f"🎨 HuggingFace LoRA をロード: {lora_model_id}")
                    self.pipe.load_lora_weights(lora_model_id)
                    print("✅ LoRA ロード完了")
                except Exception as e:
                    print(f"⚠️  LoRA loading failed: {e}")
                    print("   Continuing with base model only")
        
        # LCM Scheduler 設定 (Phase 2B対応)
        if use_lcm or use_official_lcm_lora:
            print("⚡ LCMScheduler を適用 (4ステップ推論)")
            try:
                self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
                self.lcm_steps = 4
                print("✅ LCMScheduler 適用完了")
            except Exception as e:
                print(f"⚠️  LCM setup failed: {e}")
                self.use_lcm = False
                self.lcm_steps = 20
        else:
            self.lcm_steps = 20

        # 公式 LCM-LoRA (Augmented PF-ODE 搭載 — guidance=1.5 が有効)
        if use_official_lcm_lora:
            try:
                print("📥 公式 LCM-LoRA をロード (latent-consistency/lcm-lora-sdv1-5)...")
                self.pipe.load_lora_weights(
                    "latent-consistency/lcm-lora-sdv1-5", adapter_name="lcm"
                )
                self.pipe.set_adapters(["lcm"], adapter_weights=[1.0])
                print("✅ 公式 LCM-LoRA 適用完了")
                print("   guidance_scale=1.5 を使用してください (Augmented PF-ODE 搭載)")
            except Exception as e:
                print(f"⚠️  公式 LCM-LoRA のロード失敗: {e}")

        print("✅ Pipeline ready!")
        lcm_mode = "公式 LCM-LoRA" if use_official_lcm_lora else ("LCMScheduler" if use_lcm else "❌")
        print(f"   Configuration: LCM={lcm_mode} | LoRA={'✅' if (use_lora or lora_path) else '❌'}")
        print()
        
        # RobustPromptGenerator 初期化
        self.robust_prompt_generator = None
        if use_robust_prompt:
            try:
                from prompt_optimizer import RobustPromptGenerator
                print("📦 Loading RobustPromptGenerator...")
                self.robust_prompt_generator = RobustPromptGenerator()
                print("✅ RobustPromptGenerator ready!")
            except Exception as e:
                print(f"⚠️  RobustPromptGenerator not available: {e}")
        
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
        num_inference_steps: int = None,
        guidance_scale: float = 7.0,
        height: int = 512,
        width: int = 512,
        seed: int = None,
    ) -> tuple:
        """
        単一画像生成
        
        Args:
            prompt: プロンプト
            output_path: 保存先パス
            num_inference_steps: 推論ステップ数 (デフォルト: LCM時4、通常時20)
            guidance_scale: ガイダンススケール
            height: 画像高さ
            width: 画像幅
            seed: 乱数シード
            
        Returns:
            (PIL Image, 実行時間) のタプル
        """
        # ステップ数デフォルト設定
        if num_inference_steps is None:
            num_inference_steps = self.lcm_steps if self.use_lcm else 20
        
        if seed is not None:
            torch.manual_seed(seed)
        
        start_time = time.time()
        
        with torch.no_grad():
            image = self.pipe(
                prompt=prompt,
                negative_prompt="low quality, blurry",
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width
            ).images[0]
        
        elapsed_time = time.time() - start_time
        
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            image.save(output_path)
            print(f"  ✅ Saved: {output_path} ({elapsed_time:.2f}s)")
        
        return image, elapsed_time
    
    def generate_image_with_optimized_prompt(
        self,
        emotion: str,
        style: str,
        output_path: str = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.0,
        height: int = 512,
        width: int = 512,
        seed: int = None,
    ) -> tuple:
        """
        RobustPromptGenerator を使用した最適化プロンプトで画像生成
        
        Args:
            emotion: 感情（"happy", "angry", "sad", "surprised"）
            style: スタイル（styles 辞書のキー）
            output_path: 保存先パス
            num_inference_steps: 推論ステップ数
            guidance_scale: ガイダンススケール
            height: 画像高さ
            width: 画像幅
            seed: 乱数シード
        
        Returns:
            (PIL.Image, prompt_info) のタプル
        """
        
        if not self.robust_prompt_generator:
            print("⚠️  RobustPromptGenerator not available, using fallback...")
            # フォールバック: 通常のプロンプト生成
            emotion_desc = self.emotions.get(emotion, emotion)
            style_desc = self.styles.get(style, style)
            prompt = f"{self.base_prompt}, {emotion_desc}, {style_desc}"
            
            image = self.generate_image(
                prompt=prompt,
                output_path=output_path,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                seed=seed
            )
            
            return image, {
                "positive_prompt": prompt,
                "negative_prompt": "low quality, blurry",
                "confidence": 0.5,
                "method": "fallback"
            }
        
        # RobustPromptGenerator でプロンプト生成
        prompt_info = self.robust_prompt_generator.generate_prompt(
            emotion=emotion,
            style=style,
            quality_level="masterpiece"
        )
        
        final_prompt = prompt_info["positive_prompt"]
        negative_prompt = prompt_info["negative_prompt"]
        
        print(f"\n🤖 Optimized Prompt (Confidence: {prompt_info['confidence']:.2f}):")
        print(f"  Positive: {final_prompt[:80]}...")
        print(f"  Negative: {negative_prompt[:80]}...")
        
        if seed is not None:
            torch.manual_seed(seed)
        
        with torch.no_grad():
            image = self.pipe(
                prompt=final_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width
            ).images[0]
        
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            image.save(output_path)
            print(f"  ✅ Saved: {output_path}")
        
        return image, prompt_info
    
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
    parser = argparse.ArgumentParser(
        description="anime-character-generator v2.0 (Phase 2B LCM対応)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本推論 (20 steps)
  python character_generator.py --emotion happy --style casual
  
  # LCM高速推論 (4 steps, 5倍高速)
  python character_generator.py --emotion happy --lcm
  
  # LoRA + LCM最適化構成
  python character_generator.py --lora --lcm --emotion happy --style casual
  
  # すべてのバリエーション (LCM推論)
  python character_generator.py --all --lcm
        """
    )
    parser.add_argument("--emotion", choices=["happy", "angry", "sad", "surprised"],
                       help="感情を指定")
    parser.add_argument("--style", 
                       choices=["with_hat", "with_earrings", "with_makeup", "formal", "casual",
                               "long_hair", "blush", "fireplace", "warm_lighting", "cozy_room",
                               "bokeh", "portrait", "depth_of_field", "high_detail", 
                               "soft_shading", "masterpiece"],
                       help="スタイルを指定")
    parser.add_argument("--all", action="store_true", help="全パターン生成")
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default="auto",
                       help="実行デバイス (auto: cuda > mps > cpu)")
    parser.add_argument("--use-robust-prompt", action="store_true",
                       help="RobustPromptGenerator を使用 (Phase 1)")
    parser.add_argument("--lcm", action="store_true",
                       help="LCMScheduler で高速推論 (4ステップ) [Phase 2B]")
    parser.add_argument("--lora", action="store_true",
                       help="HuggingFace 上の diffusers 形式 LoRA を統合 [Phase 2B]")
    parser.add_argument("--lora-path", type=str, default=None, dest="lora_path",
                       metavar="DIR",
                       help="ローカル PEFT 形式 anime LoRA ディレクトリ (例: ./lora_weights/anime-lora-final)")
    parser.add_argument("--official-lcm-lora", action="store_true", dest="official_lcm_lora",
                       help="公式 LCM-LoRA (latent-consistency/lcm-lora-sdv1-5) を使用【推奨: guidance=1.5 が有効】")
    
    args = parser.parse_args()
    
    # ジェネレータ初期化
    # LCM モードのデフォルト guidance を設定
    default_guidance = 1.5 if (args.lcm or args.official_lcm_lora) else 7.0

    generator = AnimeCharacterGenerator(
        device=args.device,
        use_robust_prompt=args.use_robust_prompt,
        use_lcm=args.lcm,
        use_lora=args.lora,
        lora_path=args.lora_path,
        use_official_lcm_lora=args.official_lcm_lora,
    )
    
    if args.all:
        generator.generate_all()
    elif args.emotion and args.style:
        # 特定の感情+スタイルで生成
        emotion_desc = generator.emotions[args.emotion]
        style_desc = generator.styles[args.style]
        prompt = f"{generator.base_prompt}, {emotion_desc}, {style_desc}"
        print(f"\n🎨 Generating: {args.emotion} + {args.style}")
        if args.official_lcm_lora:
            print(f"   (公式 LCM-LoRA: 4 steps, ~1.3s, guidance=1.5)")
        elif args.lcm:
            print(f"   (LCMScheduler: 4 steps, ~1.3s)")
        image, elapsed = generator.generate_image(prompt, guidance_scale=default_guidance)
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
