# Phase 3: マルチモーダル操作（Image-to-Image + ControlNet）実装ガイド

**対象フェーズ**: Phase 3 (マルチモーダル拡張)  
**推定期間**: 5-7日  
**基盤技術**: Image-to-Image パイプライン + ControlNet (Llama-2 Vision backbone)  
**依存**: Phase 2A の LoRA モデル + 任意で Phase 2B の LCM  
**成果物**: 双方向変換パイプライン + スタイル転送エンジン

---

## 📖 背景：なぜマルチモーダルか？

### 問題: 単一モード（テキスト→画像）の制限

Phase 2A までは **テキストプロンプト → 画像生成** の単一方向：

```
利用シーン:
  ✅ 「天使のキャラクターを描いて」→ 生成
  ❌ 「このスケッチをアニメ風に変換して」
  ❌ 「この写真のポーズで新しいキャラを生成」
  ❌ 「ここだけスタイルを変更して」
```

### 解決策: マルチモーダルパイプライン

Phase 3 で実装する 3 つの新機能：

```
1. Image-to-Image (I2I)
   スケッチ / 低品質画像 → アニメ風変換
   例: 手描きスケッチ → 高品質アニメキャラ

2. ControlNet
   ポーズ / 構図 → キャラクター生成
   例: ポーズ画像 → そのポーズでアニメキャラ生成

3. Inpainting
   局所編集: 髪色変更、セーター色変更など
   例: 既存画像の領域 → 新しい要素に置換
```

---

## 🎯 理論的基礎

### 1. Image-to-Image (I2I) のの仕組み

```
通常パイプライン:
  noise_latent → [100 step] → image

I2I パイプライン:
  input_image → VAE encode → image_latent
  image_latent + noise → [50 step] → image
  
  strength パラメータ:
    strength=0.0  → 元画像のまま
    strength=0.5  → 元画像とノイズを 50% ずつミックス
    strength=1.0  → 完全ノイズから再生成
```

### 2. ControlNet の概念

```
通常 Diffusion:
  noise → [UNet] → image
  条件 (テキスト) のみ

ControlNet:
  conditioning_image (ポーズ / エッジ)
         ↓
    [ControlNet エンコーダー]
         ↓
  control_embedding
         ↓
  noise + control_embedding → [UNet] → image
  
結果: テキスト条件 + 構図条件で生成
```

### 3. Inpainting（局所編集）

```
通常:
  input_image → [100 step] → output_image (全体変更)

Inpainting:
  input_image + mask → VAE encode (masked region のみノイズ化)
                    → [50 step]
                    → output_image (masked region のみ変更)

マスク:
  255 (白): 編集対象の領域
  0 (黒): 保持する領域
```

---

## 🛠️ 実装ステップ

### Step 1: 先行知識・パッケージ

```bash
# ControlNet 対応の diffusers が必要
pip install -q diffusers>=0.21.0  # >= 0.24.0 推奨
pip install -q controlnet-aux  # ControlNet 前処理

# バージョン確認
python -c "from diffusers import StableDiffusionControlNetPipeline; print('✅')"
```

### Step 2: MultimodalPipeline クラス実装

**ファイル**: `multimodal_pipeline.py` を新規作成

```python
#!/usr/bin/env python3
"""
Multimodal Pipeline: Image-to-Image + ControlNet + Inpainting

使用例:
    pipeline = MultimodalPipeline(
        lora_path="./lora_weights/anime-lora-final"
    )
    
    # Image-to-Image
    output = pipeline.image_to_image(
        input_image="sketch.png",
        prompt="anime girl",
        strength=0.8
    )
    
    # ControlNet (ポーズ)
    output = pipeline.generate_with_pose(
        pose_image="person_pose.jpg",
        prompt="anime girl"
    )
    
    # Inpainting
    output = pipeline.inpaint(
        input_image="character.png",
        mask_image="mask.png",
        prompt="blue hair"
    )
"""

import torch
import os
from pathlib import Path
from typing import Optional, Union, List
from PIL import Image
import numpy as np
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
    DDPMScheduler
)
from controlnet_aux import OpenposeDetector, CannyEdgeDetector


class MultimodalPipeline:
    """
    マルチモーダル生成パイプライン
    
    テキスト→画像、画像→画像、ControlNet による
    複数の生成方式をサポート
    """
    
    def __init__(
        self,
        base_model: str = "runwayml/stable-diffusion-v1-5",
        lora_path: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        use_lcm: bool = False
    ):
        """
        初期化
        
        Args:
            base_model: ベースモデル
            lora_path: LoRA パス（オプション）
            device: デバイス
            dtype: データ型
            use_lcm: LCM を使用するか
        """
        
        self.device = device
        self.dtype = dtype
        self.use_lcm = use_lcm
        
        print("📦 Loading base model")
        
        # テキスト→画像 (T2I)
        self.t2i_pipe = StableDiffusionPipeline.from_pretrained(
            base_model,
            torch_dtype=dtype,
            safety_checker=None
        ).to(device)
        
        # 画像→画像 (I2I)
        self.i2i_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            base_model,
            torch_dtype=dtype,
            safety_checker=None
        ).to(device)
        
        # Inpainting
        self.inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            base_model,
            torch_dtype=dtype,
            safety_checker=None
        ).to(device)
        
        # LoRA 統合
        if lora_path:
            print(f"📚 Loading LoRA: {lora_path}")
            self.t2i_pipe.unet.load_adapter(lora_path)
            self.i2i_pipe.unet.load_adapter(lora_path)
            self.inpaint_pipe.unet.load_adapter(lora_path)
        
        # ControlNet（複数タイプ対応）
        self.controlnets = {}
        self._setup_controlnets()
        
        print("✅ Multimodal pipelines ready")
    
    def _setup_controlnets(self):
        """ControlNet モデルのセットアップ"""
        
        try:
            # Canny エッジ検出
            self.controlnets["canny"] = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny",
                torch_dtype=self.dtype
            ).to(self.device)
            print("  ✓ Canny ControlNet loaded")
        except Exception as e:
            print(f"  ⚠️  Canny ControlNet error: {e}")
        
        try:
            # OpenPose
            self.controlnets["openpose"] = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-openpose",
                torch_dtype=self.dtype
            ).to(self.device)
            print("  ✓ OpenPose ControlNet loaded")
        except Exception as e:
            print(f"  ⚠️  OpenPose ControlNet error: {e}")
        
        try:
            # Depth
            self.controlnets["depth"] = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-depth",
                torch_dtype=self.dtype
            ).to(self.device)
            print("  ✓ Depth ControlNet loaded")
        except Exception as e:
            print(f"  ⚠️  Depth ControlNet error: {e}")
    
    def load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """画像を読み込んで 512x512 にリサイズ"""
        
        image = Image.open(image_path).convert("RGB")
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
        return image
    
    # ============ Mode 1: Image-to-Image ============
    
    def image_to_image(
        self,
        input_image: Union[str, Image.Image],
        prompt: str,
        negative_prompt: str = "",
        strength: float = 0.8,
        num_inference_steps: int = None,
        guidance_scale: float = 7.5
    ) -> Image.Image:
        """
        Image-to-Image 変換
        
        用途:
          - スケッチ → 高品質画像
          - 低品質画像 → アニメ風
          - スタイル転送
        
        Args:
            input_image: 入力画像
            prompt: プロンプト
            negative_prompt: ネガティブプロンプト
            strength: 変更度（0=元のまま, 1=完全再生成）
            num_inference_steps: ステップ数
            guidance_scale: 条件ガイドの強さ
        
        Returns:
            変換済み画像
        """
        
        if isinstance(input_image, str):
            input_image = self.load_image(input_image)
        
        # LCM の場合
        if self.use_lcm:
            num_inference_steps = num_inference_steps or 4
        else:
            num_inference_steps = num_inference_steps or 30
        
        print(f"🎨 Image-to-Image")
        print(f"   Strength: {strength:.1f} (0=保持, 1=再生成)")
        print(f"   Steps: {num_inference_steps}")
        
        with torch.no_grad():
            output = self.i2i_pipe(
                prompt=prompt,
                image=input_image,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt
            ).images[0]
        
        return output
    
    # ============ Mode 2: ControlNet (Pose) ============
    
    def generate_with_pose(
        self,
        pose_image: Union[str, Image.Image],
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = None,
        guidance_scale: float = 7.5
    ) -> Image.Image:
        """
        ポーズ条件付き生成
        
        用途:
          - 人物ポーズ画像 → そのポーズのアニメキャラ
          - ポーズライブラリから新規生成
        
        Args:
            pose_image: ポーズ画像（OpenPose で検出可能）
            prompt: プロンプト
            negative_prompt: ネガティブプロンプト
            num_inference_steps: ステップ数
            guidance_scale: ガイドスケール
        
        Returns:
            生成画像
        """
        
        if "openpose" not in self.controlnets:
            raise ValueError("OpenPose ControlNet not loaded")
        
        if isinstance(pose_image, str):
            pose_image = self.load_image(pose_image)
        
        # OpenPose 検出
        print("🕵️ Detecting pose with OpenPose")
        detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        pose_detected = detector(pose_image)
        
        # ControlNet パイプライン
        if self.use_lcm:
            num_inference_steps = num_inference_steps or 4
        else:
            num_inference_steps = num_inference_steps or 20
        
        print(f"🎭 ControlNet (Pose)")
        print(f"   Steps: {num_inference_steps}")
        
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnets["openpose"],
            torch_dtype=self.dtype,
            safety_checker=None
        ).to(self.device)
        
        # LoRA 統合
        # pipe.unet.load_adapter(lora_path)  # 必要に応じて
        
        with torch.no_grad():
            output = pipe(
                prompt=prompt,
                image=pose_detected,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt
            ).images[0]
        
        return output
    
    # ============ Mode 3: ControlNet (Edge) ============
    
    def generate_with_edges(
        self,
        edge_image: Union[str, Image.Image],
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = None,
        guidance_scale: float = 7.5
    ) -> Image.Image:
        """
        エッジ条件付き生成（構図指定）
        
        用途:
          - スケッチ （エッジ） → 完成画像
          - 構図指定生成
        
        Args:
            edge_image: エッジ画像
            prompt: プロンプト
            negative_prompt: ネガティブプロンプト
            num_inference_steps: ステップ数
            guidance_scale: ガイドスケール
        
        Returns:
            生成画像
        """
        
        if "canny" not in self.controlnets:
            raise ValueError("Canny ControlNet not loaded")
        
        if isinstance(edge_image, str):
            edge_image = self.load_image(edge_image)
        
        # エッジ検出
        print("📐 Detecting edges with Canny")
        detector = CannyEdgeDetector()
        edges = detector(edge_image)
        
        if self.use_lcm:
            num_inference_steps = num_inference_steps or 4
        else:
            num_inference_steps = num_inference_steps or 20
        
        print(f"📏 ControlNet (Edges)")
        print(f"   Steps: {num_inference_steps}")
        
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnets["canny"],
            torch_dtype=self.dtype,
            safety_checker=None
        ).to(self.device)
        
        with torch.no_grad():
            output = pipe(
                prompt=prompt,
                image=edges,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt
            ).images[0]
        
        return output
    
    # ============ Mode 4: Inpainting (局所編集) ============
    
    def inpaint(
        self,
        input_image: Union[str, Image.Image],
        mask_image: Union[str, Image.Image],
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = None,
        guidance_scale: float = 7.5,
        strength: float = 0.8
    ) -> Image.Image:
        """
        局所編集（Inpainting）
        
        用途:
          - 髪色変更
          - 服装変更
          - 背景変更
          - オブジェクト置換
        
        Args:
            input_image: 入力画像
            mask_image: マスク画像 (白=編集, 黒=保持)
            prompt: 編集内容のプロンプト
            negative_prompt: ネガティブプロンプト
            num_inference_steps: ステップ数
            guidance_scale: ガイドスケール
            strength: 編集の強さ
        
        Returns:
            編集済み画像
        """
        
        if isinstance(input_image, str):
            input_image = self.load_image(input_image)
        if isinstance(mask_image, str):
            mask_image = self.load_image(mask_image)
        
        # マスクを二値化（0 or 255）
        mask_array = np.array(mask_image.convert("L"))
        mask_array = (mask_array > 128).astype(np.uint8) * 255
        mask_image = Image.fromarray(mask_array)
        
        if self.use_lcm:
            num_inference_steps = num_inference_steps or 8
        else:
            num_inference_steps = num_inference_steps or 30
        
        print(f"✏️  Inpainting (局所編集)")
        print(f"   Prompt: {prompt[:50]}...")
        print(f"   Steps: {num_inference_steps}")
        
        with torch.no_grad():
            output = self.inpaint_pipe(
                prompt=prompt,
                image=input_image,
                mask_image=mask_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                strength=strength
            ).images[0]
        
        return output
    
    # ============ ユーティリティ ============
    
    def save_image(self, image: Image.Image, output_path: Union[str, Path]):
        """画像保存"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        print(f"💾 Saved to {output_path}")
    
    def batch_process(
        self,
        input_dir: str,
        output_dir: str,
        mode: str = "i2i",
        prompt: str = "anime character, masterpiece",
        **kwargs
    ):
        """
        バッチ処理
        
        Args:
            input_dir: 入力ディレクトリ
            output_dir: 出力ディレクトリ
            mode: 処理モード ("i2i", "pose", "edges", "inpaint")
            prompt: プロンプト
            **kwargs: モード固有のパラメータ
        """
        
        input_path = Path(input_dir)
        image_files = list(input_path.glob("*.png")) + list(input_path.glob("*.jpg"))
        
        print(f"🔄 Batch processing {len(image_files)} images")
        
        for i, image_file in enumerate(image_files, 1):
            print(f"\n [{i}/{len(image_files)}] Processing {image_file.name}")
            
            try:
                if mode == "i2i":
                    output = self.image_to_image(image_file, prompt, **kwargs)
                elif mode == "pose":
                    output = self.generate_with_pose(image_file, prompt, **kwargs)
                elif mode == "edges":
                    output = self.generate_with_edges(image_file, prompt, **kwargs)
                else:
                    print(f"⚠️  Unknown mode: {mode}")
                    continue
                
                output_file = Path(output_dir) / image_file.name
                self.save_image(output, output_file)
            
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """デモ実行"""
    
    pipeline = MultimodalPipeline(
        lora_path="./lora_weights/anime-lora-final",
        use_lcm=False
    )
    
    # 1. Image-to-Image
    sketch = "samples/sketch.png"
    output_i2i = pipeline.image_to_image(
        input_image=sketch,
        prompt="beautiful anime girl, long hair, detailed",
        strength=0.7
    )
    pipeline.save_image(output_i2i, "outputs/i2i_result.png")
    
    # 2. ControlNet (Pose)
    pose_image = "samples/pose.jpg"
    output_pose = pipeline.generate_with_pose(
        pose_image=pose_image,
        prompt="anime girl, standing pose, beautiful"
    )
    pipeline.save_image(output_pose, "outputs/pose_result.png")
    
    # 3. Inpainting
    character = "outputs/character.png"
    mask = "outputs/hair_mask.png"
    output_inpaint = pipeline.inpaint(
        input_image=character,
        mask_image=mask,
        prompt="blue hair, long hair"
    )
    pipeline.save_image(output_inpaint, "outputs/inpaint_result.png")


if __name__ == "__main__":
    main()
```

---

## 💡 実用例

### 例 1: スケッチ → 高品質アニメキャラ

```python
pipeline = MultimodalPipeline(lora_path="./lora_weights/anime-lora-final")

# ユーザーが手書きスケッチ
user_sketch = "user_input/sketch.png"

# スケッチをアニメ化
output = pipeline.image_to_image(
    input_image=user_sketch,
    prompt="beautiful girl, anime style, long hair",
    strength=0.9  # ほぼ再生成（スケッチ尊重）
)

output.save("results/anime_version.png")
```

### 例 2: ポーズ + プロンプト → キャラ生成

```python
# ユーザーが選んだポーズ画像
pose_reference = "references/sitting_pose.jpg"

# そのポーズでアニメキャラ生成
character = pipeline.generate_with_pose(
    pose_image=pose_reference,
    prompt="anime girl, pink hair, magical girl costume"
)

character.save("results/character_with_pose.png")
```

### 例 3: 局所編集（髪色変更）

```python
# 既存キャラ画像
original = "gallery/character_v1.png"

# 髪色変更用マスク作成
mask = create_hair_mask(original)  # 別途関数

# 髪色を青に変更
modified = pipeline.inpaint(
    input_image=original,
    mask_image=mask,
    prompt="blue hair, anime style"
)

modified.save("results/character_blue_hair.png")
```

---

## 📊 期待される成果

| 機能 | 入力 | 処理時間 | 活用 |
|-----|------|--------|------|
| Image-to-Image | スケッチ | 5-10秒 | ユーザースケッチ→高品質化 |
| ControlNet Pose | ポーズ画像 | 8-15秒 | ポーズ指定キャラ生成 |
| ControlNet Edge | エッジ | 5-10秒 | 構図指定生成 |
| Inpainting | 既存画+領域 | 3-8秒 | 局所編集・色変更 |

---

## ✅ 完了チェックリスト

- [ ] `multimodal_pipeline.py` 実装完了
- [ ] ControlNet モデルのダウンロード確認
- [ ] Image-to-Image テスト実行
- [ ] ControlNet (Pose) テスト実行
- [ ] ControlNet (Edges) テスト実行
- [ ] Inpainting テスト実行
- [ ] バッチ処理テスト
- [ ] ブログ記事執筆「ControlNet で自由度の高い生成」

---

**次のステップ**: Phase 3 完了後、[PHASE_4_DEPLOYMENT.md](PHASE_4_DEPLOYMENT.md) へ

