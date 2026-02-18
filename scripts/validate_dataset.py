#!/usr/bin/env python3
"""
LoRA 学習データセットの検証スクリプト

使用例:
    python validate_dataset.py
    python validate_dataset.py --data-dir training_data
"""

import os
from PIL import Image
from pathlib import Path
import argparse
import json


def validate_training_data(data_dir="training_data"):
    """学習データの有効性チェック"""
    
    print("="*60)
    print("📊 Training Dataset Validation")
    print("="*60)
    print(f"\n📁 Checking: {data_dir}/\n")
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ Error: Directory not found: {data_dir}")
        return False
    
    total_images = 0
    total_size = 0
    issues = []
    style_stats = {}
    resolution_stats = {}
    
    for style_dir in sorted(data_path.iterdir()):
        if not style_dir.is_dir() or style_dir.name.startswith("."):
            continue
        
        if style_dir.name in ["images", "outputs"]:
            continue
        
        print(f"📁 {style_dir.name}/")
        style_count = 0
        style_size = 0
        min_res = (float('inf'), float('inf'))
        max_res = (0, 0)
        
        for img_file in sorted(style_dir.glob("*.png")):
            try:
                img = Image.open(img_file)
                width, height = img.size
                file_size = img_file.stat().st_size / (1024 * 1024)  # MB
                
                # 解像度チェック
                if width < 256 or height < 256:
                    issues.append(f"⚠️  Small image: {img_file.name} ({width}x{height})")
                
                if width > 2000 or height > 2000:
                    issues.append(f"⚠️  Large image: {img_file.name} ({width}x{height})")
                
                min_res = (min(min_res[0], width), min(min_res[1], height))
                max_res = (max(max_res[0], width), max(max_res[1], height))
                
                # 形式チェック
                if img.format != "PNG":
                    issues.append(f"⚠️  Not PNG: {img_file.name} ({img.format})")
                
                style_count += 1
                style_size += file_size
                total_images += 1
                total_size += file_size
                
            except Exception as e:
                issues.append(f"❌ Corrupt: {img_file.name} - {str(e)[:50]}")
        
        if style_count > 0:
            style_stats[style_dir.name] = style_count
            resolution_stats[style_dir.name] = {
                "min": min_res,
                "max": max_res,
                "count": style_count,
                "size_mb": round(style_size, 2)
            }
            print(f"   ✅ {style_count} images ({round(style_size, 2)} MB)")
            print(f"      Resolution: {min_res} - {max_res}")
        else:
            print(f"   ⚠️  No images found")
    
    # メタデータ確認
    metadata_path = data_path / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            print(f"\n✅ Metadata file exists (entries: {len(metadata.get('training_data', []))})")
        except Exception as e:
            print(f"⚠️  Metadata file error: {e}")
    else:
        print(f"⚠️  No metadata.json found")
    
    # ログ確認
    log_path = data_path / "download_log.txt"
    if log_path.exists():
        print(f"✅ Log file exists")
    
    # サマリー
    print("\n" + "="*60)
    print(f"📊 Summary:")
    print(f"   Total images: {total_images}")
    print(f"   Total size: {round(total_size, 2)} MB")
    
    print(f"\n🎨 Styles:")
    for style, count in sorted(style_stats.items()):
        print(f"   {style}: {count} images")
    
    # 問題の表示
    if issues:
        print(f"\n⚠️  Issues found: {len(issues)}")
        for issue in issues[:20]:  # 最初の20件表示
            print(f"   {issue}")
    else:
        print(f"\n✅ No issues found!")
    
    # 推奨値チェック
    print(f"\n🎯 Recommendations:")
    if total_images >= 200:
        print(f"   ✅ Image count is sufficient for training")
    else:
        print(f"   ⚠️  Minimum 200 images recommended (current: {total_images})")
    
    if total_size >= 1000:
        print(f"   ✅ Dataset size is good")
    else:
        print(f"   ⚠️  Consider more images (current: {round(total_size, 2)} MB)")
    
    # 解像度チェック
    avg_widths = [stat["max"][0] for stat in resolution_stats.values()]
    if avg_widths and sum(avg_widths) / len(avg_widths) >= 512:
        print(f"   ✅ Resolution is adequate for training")
    else:
        print(f"   ⚠️  Consider images closer to 512x512")
    
    print("\n" + "="*60)
    
    is_valid = total_images >= 200 and len(issues) < 5
    return is_valid


def main():
    parser = argparse.ArgumentParser(description="Training dataset validator")
    parser.add_argument(
        "--data-dir",
        default="training_data",
        help="Training data directory (default: training_data)"
    )
    
    args = parser.parse_args()
    
    try:
        is_valid = validate_training_data(args.data_dir)
        print(f"\n{'✅ Ready for training!' if is_valid else '❌ Dataset needs improvements'}")
        return 0 if is_valid else 1
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
