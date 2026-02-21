#!/usr/bin/env bash
# HuggingFace Hub へのアップロード例
# このスクリプトはアップロード前にトークンを設定します

# 1. 環境変数の確認
if [ -z "$HF_TOKEN" ]; then
    echo "❌ HF_TOKEN が設定されていません"
    echo ""
    echo "以下の方法で HF_TOKEN を設定してください:"
    echo "  方法1: 環境変数で設定"
    echo "    export HF_TOKEN='hf_xxxxxxxxxxxxx'"
    echo ""
    echo "  方法2: HuggingFace CLI でログイン"
    echo "    huggingface-cli login"
    echo ""
    echo "  方法3: コマンドラインで直接指定"
    echo "    python upload_to_huggingface.py --hf-token 'hf_xxxxxxxxxxxxx' ..."
    echo ""
    echo "トークンは https://huggingface.co/settings/tokens で生成できます"
    exit 1
fi

# 2. Python スクリプトの実行
echo "🚀 HuggingFace Hub へアップロード開始..."
echo ""

# デフォルトパスでアップロード（カスタマイズ可能）
LORA_PATH="${1:-.lora_weights}"
REPO_NAME="${2:-anime-character-lora}"
PRIVATE_FLAG="${3:---private}"

python upload_to_huggingface.py \
    --model-path "$LORA_PATH" \
    --repo-name "$REPO_NAME" \
    "$PRIVATE_FLAG"

echo ""
echo "✅ アップロード完了"
