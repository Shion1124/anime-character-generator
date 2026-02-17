"""
anime-character-generator: 簡易版
Stable Diffusion + PyTorch (MPS/CPU) を使用した
アニメキャラクター自動生成スクリプト

Day 1-2 簡易実装版
"""

from diffusers import StableDiffusionPipeline
import torch
from pathlib import Path
from datetime import datetime
import sys


def test_mps_availability():
    """MPS デバイスの利用可能性を確認"""
    print("="*60)
    print("🔍 デバイス確認")
    print("="*60)
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    print(f"✓ MPS available: {torch.backends.mps.is_available()}")
    
    if torch.backends.mps.is_available():
        device = "mps"
        print(f"✓ Using device: MPS (Apple Silicon)")
    else:
        device = "cpu"
        print(f"✓ Using device: CPU (フォールバック)")
    
    return device


def generate_simple(device="mps"):
    """
    簡易的なアニメキャラクター生成
    """
    print("\n" + "="*60)
    print("📦 モデルロード中...")
    print("="*60)
    
    # モデルロード
    model_id = "runwayml/stable-diffusion-v1-5"
    print(f"✓ Model: {model_id}")
    
    # 自動デバイス選択
    if device == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype
        )
        pipe = pipe.to(device)
        pipe.enable_attention_slicing()  # メモリ最適化
        
        print(f"✅ モデルロード完了 (device: {device})")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        sys.exit(1)
    
    # 出力ディレクトリ作成
    output_dir = Path("outputs/simple") / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("🎨 画像生成開始")
    print("="*60)
    
    # テストプロンプト
    prompts = [
        {
            "name": "happy",
            "positive": "1girl, anime character, happy smile, cheerful, cute face, high quality",
            "negative": "low quality, blurry, dark"
        },
        {
            "name": "serious",
            "positive": "1girl, anime character, serious expression, intense eyes, beautiful, high quality",
            "negative": "low quality, blurry"
        },
        {
            "name": "sad",
            "positive": "1girl, anime character, sad expression, melancholic, teary, high quality",
            "negative": "low quality, blurry, happy"
        },
    ]
    
    for i, prompt_dict in enumerate(prompts, 1):
        name = prompt_dict["name"]
        positive = prompt_dict["positive"]
        negative = prompt_dict["negative"]
        
        print(f"\n[{i}/{len(prompts)}] 生成中: {name}")
        print(f"  Positive: {positive}")
        print(f"  Negative: {negative}")
        
        try:
            with torch.no_grad():
                result = pipe(
                    prompt=positive,
                    negative_prompt=negative,
                    num_inference_steps=20,
                    guidance_scale=7.5
                )
            
            image = result.images[0]
            
            # 保存
            filename = f"character_{name}.png"
            filepath = output_dir / filename
            image.save(filepath)
            
            print(f"  ✅ 保存: {filename}")
            
        except Exception as e:
            print(f"  ⚠️  エラー: {e}")
    
    print("\n" + "="*60)
    print(f"✅ 完了! 出力先: {output_dir}")
    print("="*60)
    
    return output_dir


def main():
    """メイン処理"""
    print("\n")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║  anime-character-generator (簡易版)                       ║")
    print("║  Stable Diffusion + PyTorch (MPS/CPU)                     ║")
    print("╚════════════════════════════════════════════════════════════╝\n")
    
    # デバイス確認
    device = test_mps_availability()
    
    # 生成実行
    output_dir = generate_simple(device=device)
    
    print("\n🎉 すべての処理が完了しました!")
    print(f"💾 生成された画像: {output_dir}")


if __name__ == "__main__":
    main()
