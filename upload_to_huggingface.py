#!/usr/bin/env python3
"""
HuggingFace Hub にLoRAモデルをアップロードするスクリプト

使用方法:
    python upload_to_huggingface.py \\
        --model-path ./anime-lora-weights \\
        --repo-name anime-character-lora \\
        --hf-token YOUR_HF_TOKEN \\
        --private False

詳細は `python upload_to_huggingface.py --help` で確認してください
"""

import argparse
import os
import json
from pathlib import Path
from typing import Optional
import sys

try:
    from huggingface_hub import (
        HfApi,
        Repository,
        create_repo,
        upload_folder,
        hf_hub_download,
    )
    from huggingface_hub.utils import RepositoryNotFoundError
except ImportError:
    print(
        "❌ huggingface_hub が見つかりません。"
        "以下でインストールしてください:\n"
        "  pip install huggingface-hub"
    )
    sys.exit(1)


class LoRAUploader:
    """HuggingFace Hub にLoRAモデルをアップロードするクラス"""

    def __init__(
        self,
        model_path: str,
        repo_name: str,
        hf_token: Optional[str] = None,
        private: bool = False,
        org_name: Optional[str] = None,
    ):
        """
        初期化

        Args:
            model_path: LoRA重みの保存ディレクトリ
            repo_name: HuggingFace Hub のリポジトリ名
            hf_token: HuggingFace API トークン
            private: プライベートリポジトリにするか
            org_name: オーガニゼーション名（オプション）
        """
        self.model_path = Path(model_path)
        self.repo_name = repo_name
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.private = private
        self.org_name = org_name
        self.api = HfApi(token=self.hf_token)

        # パス検証
        if not self.model_path.exists():
            raise FileNotFoundError(f"❌ モデルパスが見つかりません: {self.model_path}")

        if not self.model_path.is_dir():
            raise NotADirectoryError(f"❌ これはディレクトリではありません: {self.model_path}")

    def get_repo_id(self) -> str:
        """リポジトリ ID を取得"""
        if self.org_name:
            return f"{self.org_name}/{self.repo_name}"
        return self.repo_name

    def check_authentication(self) -> bool:
        """HuggingFace 認証を確認"""
        try:
            user_info = self.api.whoami()
            print(f"✅ 認証成功: {user_info['name']}")
            return True
        except Exception as e:
            print(f"❌ 認証エラー: {e}")
            print("HF_TOKEN環境変数を設定するか、--hf-token オプションで指定してください")
            return False

    def create_repository(self) -> str:
        """HuggingFace Hub にリポジトリを作成"""
        repo_id = self.get_repo_id()
        try:
            # リポジトリが存在するか確認
            self.api.repo_info(repo_id, repo_type="model")
            print(f"📦 リポジトリは既に存在します: {repo_id}")
            return repo_id
        except RepositoryNotFoundError:
            # リポジトリを作成
            print(f"📦 リポジトリを作成中: {repo_id}")
            repo_url = create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=self.private,
                token=self.hf_token,
                exist_ok=True,
            )
            print(f"✅ リポジトリ作成成功: {repo_url}")
            return repo_id

    def prepare_model_card(self) -> dict:
        """モデルカード情報を準備"""
        return {
            "library_name": "diffusers",
            "license": "openrail",
            "tags": [
                "stable-diffusion",
                "lora",
                "anime",
                "character-generation",
                "text-to-image",
                "diffusers",
            ],
            "base_model": "runwayml/stable-diffusion-v1-5",
        }

    def upload_files(self, repo_id: str) -> bool:
        """ファイルをアップロード"""
        try:
            print(f"📤 ファイルをアップロード中: {self.model_path} → {repo_id}")

            # モデルファイルをアップロード
            upload_folder(
                folder_path=str(self.model_path),
                repo_id=repo_id,
                repo_type="model",
                token=self.hf_token,
                commit_message="Upload LoRA weights",
            )

            print(f"✅ ファイルのアップロード成功")
            return True

        except Exception as e:
            print(f"❌ アップロードエラー: {e}")
            return False

    def upload_model_card(self, repo_id: str) -> bool:
        """モデルカード（README.md）をアップロード"""
        try:
            # ローカルの model card を確認
            model_card_path = Path("huggingface_model_card.md")
            if not model_card_path.exists():
                print("⚠️  huggingface_model_card.md が見つかりません")
                print("   デフォルトのモデルカードを生成します")
                self._create_default_model_card(repo_id)
            else:
                print(f"📄 モデルカードをアップロード中: {model_card_path}")
                with open(model_card_path, "r", encoding="utf-8") as f:
                    model_card_content = f.read()

                # README.md として HuggingFace に push
                self.api.upload_file(
                    path_or_fileobj=model_card_content.encode("utf-8"),
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    repo_type="model",
                    token=self.hf_token,
                    commit_message="Upload model card",
                )

                print(f"✅ モデルカードのアップロード成功")

            return True

        except Exception as e:
            print(f"❌ モデルカードのアップロードエラー: {e}")
            return False

    def _create_default_model_card(self, repo_id: str) -> str:
        """デフォルトのモデルカードを生成"""
        model_card = f"""---
tags:
- stable-diffusion
- lora
- anime
- character-generation
- diffusers
library_name: diffusers
license: openrail
base_model: runwayml/stable-diffusion-v1-5
---

# Anime Character LoRA

LoRA (Low-Rank Adaptation) を用いた高速アニメキャラクター生成モデルです。

## 使用方法

```python
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel

# ベースモデルをロード
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)

# LoRA 重みをロード
pipe.unet = PeftModel.from_pretrained(
    pipe.unet,
    "{repo_id}",
    adapter_name="anime_lora"
)

pipe = pipe.to("cuda")

# 画像生成
prompt = "1girl, anime character, masterpiece, high quality"
image = pipe(
    prompt=prompt,
    num_inference_steps=20,
    guidance_scale=7.5,
    height=512,
    width=512
).images[0]

image.save("output.png")
```

## モデル詳細

- **ベースモデル:** Stable Diffusion v1.5
- **LoRA ランク:** 32
- **LoRA アルファ:** 32
- **推論速度:** 20ステップで約500ms (GPU T4)

## ライセンス

OpenRAIL-M ライセンス準拠

詳細は [OpenRAIL License](https://huggingface.co/spaces/CompVis/stable-diffusion-license) を参照してください。
"""
        return model_card

    def create_model_card_metadata(self) -> dict:
        """モデルカード用のメタデータを作成"""
        return {
            "tags": [
                "stable-diffusion",
                "lora",
                "anime",
                "character-generation",
                "text-to-image",
                "diffusers",
            ],
            "library_name": "diffusers",
            "base_model": "runwayml/stable-diffusion-v1-5",
            "license": "openrail",
        }

    def upload(self) -> bool:
        """アップロード処理を実行"""
        print("=" * 60)
        print("🚀 HuggingFace Hub へのアップロード開始")
        print("=" * 60)

        # 1. 認証確認
        if not self.check_authentication():
            return False

        # 2. リポジトリ作成/確認
        try:
            repo_id = self.create_repository()
        except Exception as e:
            print(f"❌ リポジトリの作成/確認エラー: {e}")
            return False

        # 3. ファイルアップロード
        if not self.upload_files(repo_id):
            return False

        # 4. モデルカードアップロード
        if not self.upload_model_card(repo_id):
            return False

        print("=" * 60)
        print("✅ アップロード完了！")
        print("=" * 60)
        print(f"🎉 モデルは以下で利用可能です:")
        print(f"   https://huggingface.co/{repo_id}")
        print()
        print(f"📝 リポジトリ ID: {repo_id}")
        print(f"📦 モデルぱス: {self.model_path}")
        print()

        return True


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="LoRA モデルを HuggingFace Hub にアップロード",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本的な使用方法
  python upload_to_huggingface.py \\
    --model-path ./anime-lora-weights \\
    --repo-name anime-character-lora

  # プライベートリポジトリとして公開
  python upload_to_huggingface.py \\
    --model-path ./anime-lora-weights \\
    --repo-name anime-character-lora \\
    --private

  # オーガニゼーション名を指定
  python upload_to_huggingface.py \\
    --model-path ./anime-lora-weights \\
    --repo-name anime-character-lora \\
    --org-name my-organization

環境変数:
  HF_TOKEN: HuggingFace API トークン（--hf-token で上書き可能）

注意:
  - HF_TOKEN 環境変数が設定されていない場合、--hf-token を指定してください
  - トークンは https://huggingface.co/settings/tokens で生成できます
        """,
    )

    parser.add_argument(
        "--model-path",
        required=True,
        help="LoRA重みの保存ディレクトリパス",
    )

    parser.add_argument(
        "--repo-name",
        required=True,
        help="HuggingFace Hub でのリポジトリ名",
    )

    parser.add_argument(
        "--hf-token",
        help="HuggingFace API トークン（デフォルト: $HF_TOKEN）",
    )

    parser.add_argument(
        "--private",
        action="store_true",
        help="プライベートリポジトリとして作成（デフォルト: 公開）",
    )

    parser.add_argument(
        "--org-name",
        help="オーガニゼーション名（オプション）",
    )

    args = parser.parse_args()

    # アップローダー初期化
    try:
        uploader = LoRAUploader(
            model_path=args.model_path,
            repo_name=args.repo_name,
            hf_token=args.hf_token,
            private=args.private,
            org_name=args.org_name,
        )
    except Exception as e:
        print(f"❌ 初期化エラー: {e}")
        sys.exit(1)

    # アップロード実行
    success = uploader.upload()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
