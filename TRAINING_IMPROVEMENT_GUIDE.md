# LoRA トレーニング改善ガイド — Phase 2A へ

> 📝 **用途**: ブログ記事（Day3-4）の下部に追加するセクション

---

## 🎯 Phase 1.5 の実装完了！

ブログで紹介した Day3-4 の **Phase 1.5（基本実装版）は完了**しました。以下のが実装されています：

✅ Stable Diffusion v1.5 へのLoRA統合  
✅ Google Colab T4 GPU での実行（無料枠対応）  
✅ チェックポイント保存機能  
✅ トレーニングログ保存  
✅ HuggingFace Hub へのモデルカード自動生成  
✅ サンプル画像付きアップロード  

---

## 📊 トレーニング結果の分析

Phase 1.5 でのトレーニング結果（実測値例）:

```
📋 トレーニング統計
   Epochs:              10
   Batch Size:          2
   Learning Rate:       1e-4
   LoRA Rank:           8
   LoRA Alpha:          32.0
   学習時間:            約 5-7 時間（Colab T4）

📈 Loss の推移
   初期 (Epoch 1):      0.1404
   最小値 (Epoch 3):    0.1357 ✅
   最終値 (Epoch 10):   0.1461 ⚠️
   全体改善度:          -3.7%（悪化傾向）
```

### 評価

**良い点：**
- LoRA の基本実装が正常に機能 ✅
- モデルが Epoch 3 で loss 低下を達成 ✅
- Colab 無料枠で完全に実行可能 ✅

**改善の余地：**
- ⚠️ Loss 曲線が Epoch 4 で 30% 急上昇（不安定）
- ⚠️ Epoch 3 以降ロスが上昇（過学習 or LR Too High の可能性）
- ⚠️ 最終改善度が負（学習の有効性が下降）

---

## 🚀 次の段階: Phase 2A — パラメータ最適化

Phase 1.5 の結果を踏まえ、**より高品質な LoRA を得る** ための Phase 2A を提案します。

### 原因分析

1. **学習率が高すぎた** (`1e-4`)
   - Diffusion Models の LoRA は学習率に非常に敏感
   - 小規模データセット（300枚）では LR を低くする必要あり
   - Epoch 4 での急上昇がこれを示唆している

2. **LoRA ランクが小さすぎた** (`rank=8`)
   - rank が低いと表現力が制限される
   - Loss が 0.1357 で Plateau（改善停止）している

3. **データセットが最小限** (300 images)
   - イテレーション数: 300 × 10 = 3000
   - 推奨: 10,000+ イテレーション
   - 解決するには画像数を増やすか、epochs を延ばすか

4. **学習スケジューリングが基本的** (線形 LR)
   - Cosine annealing などの高度なスケジューラを試す価値あり

---

## 💡 改善案（3 段階レベル）

### 🟢 **Level 1: 最小限の改善（推奨・すぐに試す）**

**変更項目:**

| パラメータ | Phase 1.5 | Phase 2A L1 | 理由 |
|----------|-----------|------------|------|
| `EPOCHS` | 10 | **20** | イテレーション数を倍に → 学習時間充分 |
| `BATCH_SIZE` | 2 | **4** | バッチサイズ大 → 勾配推定精度向上 |
| `LEARNING_RATE` | 1e-4 | **5e-5** | 🔴 最重要！不安定性を低減 |
| `LORA_RANK` | 8 | **16** | 表現力向上 → スタイル表現が豊かに |
| `LORA_ALPHA` | 32.0 | **64.0** | rank に合わせてスケーリング |

**実装方法:**
1. ノートブック (Step 6 セル) のパラメータを上記に変更
2. **他の部分は変更せず** 再実行
3. 学習時間: +50% (5-7時間 → 7-10時間)

**期待される改善:**
- 💾 Loss 改善度: -3.7% → **+10-15%**
- 📈 損失曲線: 不安定 → **滑らかに低下**
- 🎨 推論品質: 目視確認で「洗練度」向上

**VRAM 消費:**
- Rank 8, Batch 2: ~8.0 GB
- Rank 16, Batch 4: ~8.5 GB (Colab T4 の 16.0 GB 内に収まる) ✅

---

### 🟡 **Level 2: データセット拡張**

**実装:**

```bash
# ホームマシンで実行
cd anime-character-generator

# 现在の 300 枚から 600 枚に増やす
python scripts/download_danbooru.py --limit 150  # さらに 150 枚追加

# または手動で Danbooru から 200-300 枚のアニメ画像を選別
```

**効果:**
- 多様性向上 → Overfitting 低減
- 学習が深く進む
- Loss 曲線のさらなる改善

**組み合わせ:**
Level 1 + Level 2 を同時実施した場合
- 学習時間: 10-15 時間（長いため分割実行推奨）
- 期待改善度: **+15-25%**

---

### 🔴 **Level 3: 高度な最適化（Phase 2B）**

`train_lora.py` に以下を実装:

```python
# Cosine annealing scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR

# Gradient accumulation steps
gradient_accumulation_steps = 2

# Warmup steps
warmup_steps = 500

# Gradient clipping
max_grad_norm = 1.0

# Local rank（分散学習対応）
local_rank = 0
```

**実装担当:**
- 筆者による Phase 2B 実装（別庭園で提供予定）
- 複雑度が上がるため推奨度は中程度

**期待改善度:** +20-30%

---

## 📋 実装チェックリスト

### Phase 2A Level 1 を試す場合

- [ ] 現在の `training_log.json` をバックアップ
  ```bash
  cp /content/lora_weights/training_log.json /content/lora_weights/training_log_v1.5.json
  ```

- [ ] Step 6 コードのパラメータを以下に変更
  ```python
  EPOCHS = 20
  BATCH_SIZE = 4
  LEARNING_RATE = 5e-5
  LORA_RANK = 16
  LORA_ALPHA = 64.0
  OUTPUT_DIR = "/content/lora_weights_v2a"  # v2a として別保存
  ```

- [ ] Step 6 を再実行（~10 時間）

- [ ] Step 7 でトレーニングログを確認
  - `new_training_log.json` で loss 改善度を確認
  - グラフをプロット

- [ ] 改善が見られれば、Step 11 で HuggingFace にアップロード

---

## 📊 結果比較方法

### Loss の改善度を比較

```bash
# ローカルで実行（Python）
import json

# Phase 1.5
with open("training_log_v1.5.json") as f:
    log_v1 = json.load(f)
init_loss_v1 = log_v1["losses"][0]["loss"]
final_loss_v1 = log_v1["losses"][-1]["loss"]
improvement_v1 = (init_loss_v1 - final_loss_v1) / init_loss_v1 * 100

# Phase 2A
with open("training_log_v2a.json") as f:
    log_v2a = json.load(f)
init_loss_v2a = log_v2a["losses"][0]["loss"]
final_loss_v2a = log_v2a["losses"][-1]["loss"]
improvement_v2a = (init_loss_v2a - final_loss_v2a) / init_loss_v2a * 100

print(f"Phase 1.5: {improvement_v1:.1f}% 改善")
print(f"Phase 2A:  {improvement_v2a:.1f}% 改善")
```

### 推論品質を目視確認

```python
# Step 10 推論テストで、複数プロンプトで比較
prompts = [
    "1girl, anime watercolor style, masterpiece",
    "1girl, oil painting aesthetic, soft focus",
    "1girl, sketch aesthetic, detailed face",
]

for prompt in prompts:
    image_v1 = generate_with_lora("v1.5", prompt)  # Phase 1.5 結果
    image_v2a = generate_with_lora("v2a", prompt)  # Phase 2A 結果
    
    # 目視確認: v2a の方が「洗練度」が高いか？
```

---

## 🔗 次に参考になるリソース

### Papers（学術的背景）

1. **Hu et al. (2021)** - LoRA: Low-Rank Adaptation of Large Language Models
   - https://arxiv.org/abs/2106.09685
   - LoRA の基本理論

2. **Rombach et al. (2022)** - High-Resolution Image Synthesis with Latent Diffusion Models
   - https://arxiv.org/abs/2112.10752
   - Stable Diffusion の学習戦略

3. **Luo et al. (2023)** - Latent Consistency Models
   - https://arxiv.org/abs/2310.04378
   - 高速推論への応用

### Open Source Projects

- [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft)
- [Diffusers](https://huggingface.co/docs/diffusers/)
- [Stable Diffusion Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

---

## 🎯 推奨実行順序

```
Phase 1.5 完了
    ↓
training_log_v1.5.json をバックアップ
    ↓
Step 6 パラメータを Phase 2A L1 に変更
    ↓
Step 6 を再実行（~10時間）
    ↓
Step 7 で loss 改善度を確認
    ↓
改善が見られれば、Step 11 で HF にアップロード
    ↓
【オプション】さらに改善したければ Level 2/3 へ
```

---

## ❓ よくある質問

**Q: Phase 1.5 の結果で十分では？**  
A: Phase 1.5 は「基本実装の検証」が目的。推論品質を高めるなら Phase 2A 推奨。

**Q: 学習時間が心配...**  
A: Colab は セッション 12 時間制限あり。10 時間で完了する Level 1 をお勧め。超える場合は分割実行（チェックポイント機能）で対応可能。

**Q: VRAM が足りない場合は？**  
A: `BATCH_SIZE = 2` に戻すか、`gradient_accumulation_steps = 2` で代替。

**Q: Phase 2B はいつ？**  
A: 今秋の Day5-6 ブログで予定。

---

## 📝 フィードバック

改善案の実装結果や、新しいアイデアがあればぜひ GitHub Issues でお知らせください！

- 📧 GitHub Issues: https://github.com/Shion1124/anime-character-generator/issues
- 💬 Discussions: https://github.com/Shion1124/anime-character-generator/discussions

---

**Happy Training! 🎨**

**作成日**: 2026年2月21日  
**対象**: anime-character-generator v1.5 (LoRA Edition)
