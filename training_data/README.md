# Training Data Directory

このディレクトリは LoRA ファインチューニング用の学習画像を保存します。

## 使い方

### Step 1: Danbooru から画像をダウンロード

```bash
# 試行実行（各スタイル10枚、合計50枚）
python ../scripts/download_danbooru.py --limit 10

# 本実行（各スタイル60枚、合計300枚）
python ../scripts/download_danbooru.py --limit 60 &
```

**重要**: `--limit 60` を実行するとバックグラウンドでダウンロードが進行します。
GPU メモリが必要ない場合、この間に他の処理（Step 2: train_lora.py 実装）を進めることができます。

### Step 2: ダウンロード完了確認

```bash
ls -lh training_data/
# 以下のディレクトリが作成されます：
# - impressionist_style/       (60枚)
# - soft_focus_landscape/      (60枚)
# - oil_painting_aesthetic/    (60枚)
# - sketch_aesthetic/          (60枚)
# - pastel_softness/           (60枚)
# - metadata.json             (メタデータ)
```

### Step 3: データ検証

```bash
# 画像の有効性をチェック
python ../scripts/validate_dataset.py
```

## ディレクトリ構造

```
training_data/
├── impressionist_style/           # 印象派的タッチ
│   ├── impressionist_style_000.png
│   ├── impressionist_style_001.png
│   └── ... (60枚)
├── soft_focus_landscape/          # 朧げな風景
│   ├── soft_focus_landscape_000.png
│   └── ... (60枚)
├── oil_painting_aesthetic/        # 油彩画的
│   └── ... (60枚)
├── sketch_aesthetic/              # スケッチ風
│   └── ... (60枚)
├── pastel_softness/               # パステル調
│   └── ... (60枚)
├── metadata.json                  # 画像メタデータ
└── download_log.txt               # ダウンロードログ
```

## 注意事項

⚠️ **ライセンス確認**
- Danbooru からダウンロードした画像は様々なライセンス下にあります
- HuggingFace Hub に学習データを公開する場合は、ライセンスを確認してください

⚠️ **ストレージ容量**
- 300 枚の PNG 画像で約 2-3GB 必要
- ファイルシステムに十分な空き容量を確認してください

⚠️ **ネットワーク**
- ダウンロードには高速インターネット接続が必要です
- バックグラウンド実行時はネットワーク接続を保持してください

## トラブルシューティング

### Q: ダウンロードが途中で止まった

A: `download_log.txt` でエラー詳細を確認してください。再度実行すれば、既にダウンロード済みの画像はスキップされます。

### Q: 画像が表示されない

A: `validate_dataset.py` で画像の有効性をチェック：
```bash
python ../scripts/validate_dataset.py
```

### Q: メタデータが生成されない

A: `metadata.json` が同期的に生成されます。ダウンロード完了後に確認してください。

---

**詳細**: 親ディレクトリの `dev_peft.md` を参照してください
