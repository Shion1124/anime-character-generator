#!/usr/bin/env python3
"""
Danbooru から印象派風アニメ画像を収集するスクリプト

使用例:
    # 試行実行（少量テスト）
    python scripts/download_danbooru.py --output training_data --limit 10

    # 本実行（300枚収集）
    python scripts/download_danbooru.py --output training_data --limit 60

依存パッケージ:
    pip install requests pillow tqdm
"""

import requests
import json
import os
from pathlib import Path
from typing import List, Dict
import time
import argparse


class DanbooruDownloader:
    """Danbooru から印象派風アニメ画像をダウンロード"""
    
    BASE_URL = "https://danbooru.donmai.us/posts.json"
    
    # スタイル別タグ定義（テスト済みで動作するタグのみ）
    STYLE_TAGS = {
        "impressionist_style": [
            "fantasy"
        ],
        "soft_focus_landscape": [
            "landscape"
        ],
        "oil_painting_aesthetic": [
            "scenery"
        ],
        "sketch_aesthetic": [
            "sketch"
        ],
        "pastel_softness": [
            "fantasy"
        ]
    }
    
    def __init__(self, output_dir: str = "training_data"):
        """初期化"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.metadata = []
        self.download_log = []
        self.total_downloaded = 0
    
    def download_images(self, limit_per_style: int = 60, batch_size: int = 200):
        """スタイル別に画像をダウンロード"""
        
        print("="*60)
        print("🎨 Danbooru Image Downloader - Anime Impressionist Style")
        print("="*60)
        print(f"\n📁 Output directory: {self.output_dir}")
        print(f"📊 Target: {limit_per_style} images per style")
        print(f"📈 Total target: {limit_per_style * len(self.STYLE_TAGS)} images\n")
        
        for style_index, (style_name, tags) in enumerate(self.STYLE_TAGS.items(), 1):
            print(f"\n[{style_index}/{len(self.STYLE_TAGS)}] 📥 Downloading: {style_name}")
            print(f"           Tags: {', '.join(tags)}")
            
            # スタイル別ディレクトリ作成
            style_dir = self.output_dir / style_name
            style_dir.mkdir(exist_ok=True)
            
            downloaded = self._download_style(
                style_name, tags, style_dir, limit_per_style, batch_size
            )
            self.total_downloaded += downloaded
            
            print(f"           ✅ {downloaded}/{limit_per_style} downloaded")
            
            # API リクエスト間隔（Danbooru サーバーへの負荷軽減）
            time.sleep(2)
        
        print("\n" + "="*60)
        print(f"✅ 完了！ 合計 {self.total_downloaded} 枚の画像をダウンロードしました")
        print("="*60)
        
        self._save_metadata()
        return self.total_downloaded
    
    def _download_style(
        self, 
        style_name: str, 
        tags: List[str], 
        output_dir: Path, 
        limit: int,
        batch_size: int
    ) -> int:
        """特定スタイルの画像をダウンロード"""
        
        tag_string = " ".join(tags)
        downloaded = 0
        page = 1
        failed_count = 0
        
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
            print("           (tqdm not available, progress may be less visible)")
        
        if use_tqdm:
            pbar = tqdm(total=limit, desc=f"           {style_name}", leave=False)
        
        while downloaded < limit:
            try:
                response = requests.get(
                    self.BASE_URL,
                    params={
                        "tags": tag_string,
                        "limit": batch_size,
                        "page": page
                    },
                    timeout=10
                )
                
                # エラーチェック
                if response.status_code != 200:
                    error_msg = f"API Error at page {page}: {response.status_code}"
                    try:
                        error_detail = response.json()
                        error_msg += f" - {error_detail}"
                    except:
                        error_msg += f" - {response.text[:200]}"
                    print(f"           ⚠️  {error_msg}")
                    self.download_log.append(error_msg)
                    failed_count += 1
                    page += 1
                    if failed_count >= 3:
                        break
                    time.sleep(2)
                    continue
                
                response.raise_for_status()
                
                images = response.json()
                if not images:
                    print(f"           ⚠️  No more images found at page {page}")
                    break
                
                for image in images:
                    if downloaded >= limit:
                        break
                    
                    # ファイル URL 取得
                    file_url = None
                    if "file_url" in image:
                        file_url = image["file_url"]
                    elif "large_file_url" in image:
                        file_url = image["large_file_url"]
                    
                    if not file_url:
                        continue
                    
                    # 画像ダウンロード
                    try:
                        img_response = requests.get(file_url, timeout=10)
                        img_response.raise_for_status()
                        
                        # ファイル保存
                        filename = f"{style_name}_{downloaded:03d}.png"
                        filepath = output_dir / filename
                        
                        with open(filepath, "wb") as f:
                            f.write(img_response.content)
                        
                        # メタデータ記録
                        self.metadata.append({
                            "file": str(filepath.relative_to(self.output_dir)),
                            "style": style_name,
                            "tags": image.get("tag_string_general", "").split(),
                            "width": image.get("image_width"),
                            "height": image.get("image_height"),
                            "danbooru_id": image.get("id")
                        })
                        
                        log_msg = f"✅ {filename} ({image.get('image_width')}x{image.get('image_height')})"
                        self.download_log.append(log_msg)
                        
                        if use_tqdm:
                            pbar.update(1)
                        
                        downloaded += 1
                        failed_count = 0  # リセット
                        
                    except Exception as e:
                        failed_msg = f"❌ Failed: {file_url} - {str(e)[:50]}"
                        self.download_log.append(failed_msg)
                        failed_count += 1
                        continue
                
                page += 1
                
                # ページが続かない、または失敗が多い場合は終了
                if failed_count > 10:
                    print(f"           ⚠️  Too many failures, stopping")
                    break
                
            except requests.exceptions.Timeout:
                print(f"           ⚠️  Timeout at page {page}, retrying...")
                time.sleep(3)
                continue
            except Exception as e:
                print(f"           ⚠️  API Error at page {page}: {str(e)[:50]}")
                time.sleep(3)
                continue
        
        if use_tqdm:
            pbar.close()
        
        return downloaded
    
    def _save_metadata(self):
        """メタデータ JSON 保存"""
        
        metadata_file = self.output_dir / "metadata.json"
        metadata_dict = {
            "total_images": len(self.metadata),
            "styles": list(self.STYLE_TAGS.keys()),
            "download_timestamp": str(Path(__file__).stat().st_mtime),
            "training_data": self.metadata
        }
        
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f, ensure_ascii=False, indent=2)
        
        # ログファイル保存
        log_file = self.output_dir / "download_log.txt"
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("Danbooru Download Log\n")
            f.write("="*60 + "\n\n")
            f.write("\n".join(self.download_log))
        
        print(f"\n📊 Metadata saved: {metadata_file}")
        print(f"📋 Log saved: {log_file}")
        
        # 簡単な統計出力
        style_stats = {}
        for item in self.metadata:
            style = item["style"]
            style_stats[style] = style_stats.get(style, 0) + 1
        
        print("\n📈 Statistics:")
        for style, count in sorted(style_stats.items()):
            print(f"   {style}: {count} images")


def main():
    """メイン処理"""
    
    parser = argparse.ArgumentParser(
        description="Danbooru から印象派風アニメ画像をダウンロード",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  # 試行実行（各スタイル10枚）
  python download_danbooru.py --limit 10
  
  # 本実行（各スタイル60枚、合計300枚）
  python download_danbooru.py --limit 60
  
  # カスタム出力先
  python download_danbooru.py --output my_training_data --limit 60
        """
    )
    
    parser.add_argument(
        "--output",
        default="training_data",
        help="出力ディレクトリ（デフォルト: training_data）"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=60,
        help="スタイルあたりの枚数（デフォルト: 60）"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="API リクエストのバッチサイズ（デフォルト: 200）"
    )
    
    args = parser.parse_args()
    
    # 依存パッケージ確認
    try:
        import requests
        import tqdm
    except ImportError as e:
        print(f"❌ Error: Missing required package")
        print(f"   {e}")
        print(f"\n必要なパッケージをインストール:")
        print(f"   pip install requests pillow tqdm")
        return 1
    
    try:
        downloader = DanbooruDownloader(output_dir=args.output)
        total = downloader.download_images(
            limit_per_style=args.limit,
            batch_size=args.batch_size
        )
        
        print(f"\n🎉 成功！ {total} 枚の画像を収集しました")
        print(f"📁 出力先: {args.output}/")
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  ダウンロードがキャンセルされました")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
