# 🎨 anime-character-generator - GitHub Copilot カスタムインストラクション

**バージョン**: 1.0 (2026-02-19)  
**対象**: GitHub Copilot + VS Code / Cursor / Claude Code  
**目的**: プロジェクト固有のコーディング規約・論文との整合性を確保

---

## 📖 プロジェクト概要

### プロジェクトの三層構造

このプロジェクトは、**PyTorch + Stable Diffusion** による日本アニメキャラクター自動生成システムの進化を3バージョンで管理します。

#### v1.0: 基本実装（完成）
- **説明**: PyTorch + Diffusers による基本的なアニメキャラクター生成
- **ブログ対応**: Day3-4_implementation_guide.md で完全説明
- **ファイル**: `character_generator_v1.py` + `anime_generator_colab_simple_v1.0.ipynb`
- **特徴**: 分かりやすさ、再現性、学習教材としての価値
- **速度**: 3.8秒/画像（T4 GPU）

#### v1.5: LoRA ファインチューニング版（試行版・課題あり）
- **説明**: v1.0 + LoRA による日本アニメスタイル特化版
- **ブログ対応**: Day3-4_implementation_guide.md のLoRAセクション
- **ファイル**: `character_generator_v1_lora.py` + `anime_generator_colab_lora_v1.5.ipynb`
- **既知の課題** ⚠️:
  1. テキストレベルの脆弱性（Gao et al. 2306.13103）
  2. 推論速度が遅い（3.8秒/画像）
  3. マルチモーダル入力未対応（テキストのみ）
  4. 本番環境対応なし（研究スクリプト形式）
- **役割**: 試行錯誤の過程を示す＆学習教材

#### v2.0: 学術的改善版（Phase 1-4 実装進行中）
- **説明**: 論文ベースの段階的改善を実装したプロダクション版
- **ガイド**: IMPLEMENTATION_ROADMAP.md + PHASE_[1-4]_*.md
- **ファイル**: `character_generator.py` + 複数の支援スクリプト
- **改善内容**:
  - **Phase 1**: Gemini LLM による多層冗長プロンプト（Gao対応）
  - **Phase 2A**: PEFT LoRA による効率的なファインチューニング
  - **Phase 2B**: LCM蒸留による12倍推論高速化（1秒/画像）
  - **Phase 3**: Image-to-Image + ControlNet + アニメーション生成
  - **Phase 4**: Streamlit UI + FastAPI + クラウドデプロイ

---

## 🛠️ 技術スタック

### コア依存関係

```
【基盤】
- Python 3.10+
- PyTorch (torch, torchvision)
- Diffusers (HuggingFace)
- Transformers

【バージョン別】
v1.0/v1.5:
  - PIL/Pillow        （画像処理）
  - numpy            （数値計算）
  - matplotlib       （可視化）

v2.0 Phase 1:
  + Google Generative AI SDK  （Gemini API）
  + Anthropic Claude API      （プロンプト生成）

v2.0 Phase 2A:
  + PEFT               （LoRA ファインチューニング）

v2.0 Phase 2B:
  + LCM Scheduler     （推論高速化）

v2.0 Phase 3:
  + ControlNet        （条件付き生成）

v2.0 Phase 4:
  + Streamlit         （Web UI）
  + FastAPI          （REST API）
  + Docker           （コンテナ化）
```

### 環境・実行方法

- **開発環境**: Google Colab (Tesla T4 GPU)、ローカルMac M1+
- **推奨**: Google Colab + 任意の Python IDE
- **本番**: Docker コンテナ + クラウド（GCP/Heroku/Railway）

---

## 📝 コーディング規約

### Python スタイル

#### 拡張子別
- **`.py`ファイル**: PEP 8 準拠（黒点は使用しない）
- **`.ipynb`ファイル**: セル単位での実行ブロック分割、詳細コメント

#### 命名規則

```python
# ✅ 推奨
class AnimeCharacterGenerator:              # クラス: PascalCase
    def generate_batch(self):               # メソッド: snake_case
        device = "cuda"                      # 変数: snake_case
        EMOTION_LIST = ["happy", "angry"]   # 定数: UPPER_SNAKE_CASE
        
# ❌ 避ける
class anime_character_generator:            # クラスは PascalCase
def GenerateBatch():                        # メソッドは snake_case
```

#### ドキュメンテーション

すべてのクラス・メソッドに **docstring** を記述してください。フォーマット：

```python
def generate_image(
    self,
    prompt: str,
    num_steps: int = 20,
    guidance_scale: float = 7.0
) -> Image.Image:
    """
    単一画像生成パイプライン
    
    Args:
        prompt (str): 生成対象の説明
        num_steps (int): 拡散ステップ数（推奨: 15-50）
        guidance_scale (float): Classifier-free guidance スケール（推奨: 5-15）
    
    Returns:
        Image.Image: 生成された PIL Image オブジェクト
    
    Raises:
        RuntimeError: GPU が利用不可の場合
        ValueError: prompt が空文字列の場合
    
    Example:
        >>> generator = AnimeCharacterGenerator()
        >>> img = generator.generate_image("happy anime girl")
        >>> img.save("output.png")
    
    Version History:
        v1.0: 基本実装
        v2.0: Gemini LLM統合、速度3倍向上
    """
    # ...実装
```

#### 型ヒント

**必須**: すべての関数に型ヒントを記述

```python
from typing import Dict, List, Tuple, Optional
from PIL import Image

# ✅ 良い例
def generate_batch(
    self,
    prompts: List[str],
    seeds: Optional[List[int]] = None
) -> Dict[str, Image.Image]:
    pass

# ❌ 避ける
def generate_batch(self, prompts, seeds=None):
    pass
```

### Jupyter Notebook ガイドライン

#### セル構成
1. **Markdown 1**: ノートブック全体の説明 + バージョン情報
2. **Code 1**: 環境セットアップ（GPU確認、ライブラリインポート）
3. **Markdown 2**: Step ごとの説明
4. **Code 2**: Step の実装
5. ... (繰り返し)

#### Markdown セル例

```markdown
# 🎨 anime-character-generator v1.5

## ノートブック概要
このノートブックでは、Stable Diffusion v1.5 に LoRA ファインチューニングを適用し、
日本アニメスタイルへの特化度を高めます。

**バージョン**: v1.5 (試行版・課題あり)  
**ガイド**: [Day3-4_implementation_guide.md](../)  
**所要時間**: 約 3 日 (Colab T4 GPU)

### ⚠️ 既知の課題
1. テキストレベルノイズへの脆弱性（Gao et al. 論文参照）
2. 推論速度: 3.8秒/画像
3. マルチモーダル入力未対応

### ✅ v2.0 での改善内容
これらの課題は v2.0 で Phase 1-4 にて段階的に解決されます。
詳細は IMPLEMENTATION_ROADMAP.md を参照してください。

---

## Step 1: GPU 環境確認
```

#### コード セル例

```python
# 各セルの冒頭に分かりやすいコメントを記述
# === Step 1: GPU環境確認 ===
import torch

print(f"GPU利用可能: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"デバイス: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
```

---

## 🎓 論文ベースの実装方針

### 最重要: 「論文との整合性検証」プロセス

このプロジェクトは **複数の学術論文に基づいて設計・実装** されています。AI生成時に、コード実装が論文の内容と正確に対応しているかを **必ず検証** してください。

### 論文リスト（Phase別対応）

| Phase | 論文 | 著者 | DOI | 実装内容 | 検証ポイント |
|-------|------|------|-----|--------|----------|
| **基礎** | Denoising Diffusion Probabilistic Models | Ho et al. | 2006.11239 | 拡散モデルの基本アルゴリズム | ノイズスケジュール、サンプリング方式が理論一致か |
| **基礎** | High-Resolution Image Synthesis with Latent Diffusion | Rombach et al. | 2112.10752 | Stable Diffusion v1.5 の基礎 | 潜在空間での処理、VAE エンコーダ・デコーダ |
| **Phase 1** | Evaluating Robustness of Text-to-Image Models | Gao et al. | 2306.13103 | タイポ・グリフ攻撃への脆弱性、対策手法 | 多層冗長プロンプト設計が論文提案に一致か |
| **Phase 2B** | Latent Consistency Models for Fast Image Generation | Luo et al. | 2310.04378 | LCM による推論高速化 | スケジューラ、蒸留ロス式が論文一致か |

### Phase 別の検証ポイント

#### ✅ Phase 1: LLM プロンプト最適化

**対象論文**: Gao et al. (2306.13103) "Evaluating Robustness of Text-to-Image Models against Real-world Attacks"

**論文から学ぶべき内容**:
- 文字レベルのノイズ（タイポ、グリフ変形）が生成結果に劇的に影響
- 単一の直線的プロンプトは脆弱性がある
- 解決案: 複数の言い換え、類義語による冗長性確保

**実装検証項目**:
- [ ] Gemini API で生成されるプロンプトが複数の言い換えを含んでいるか？
- [ ] 各プロンプト案の多様性が確認できるか？（同じ意味で異なるトークン使用）
- [ ] 生成スコア（信頼度）の計算ロジックが論文の提案手法に対応しているか？
- [ ] テスト: 同じプロンプトで複数回生成した際、結果の一貫性が改善されたか？

**コード検証例**:
```python
# character_generator.py で実装する場合
class RobustPromptGenerator:
    """
    Gao et al. (2306.13103) に基づく多層冗長プロンプト生成
    
    論文の提案:
    - 単一プロンプトより複数の言い換えを使用
    - 信頼度スコア導入で品質制御
    
    実装確認項目:
    ✓ prompt_variants の生成（最低 3-5 個の異なる言い換え）
    ✓ confidence_scores の計算（0-1 スケール）
    ✓ テキストノイズに対する耐性テスト
    """
```

#### ✅ Phase 2A: LoRA ファインチューニング

**対象論文**: Rombach et al. (2112.10752) + PEFT LoRA 論文

**検証項目**:
- [ ] LoRA の秩（rank）・α値がベストプラクティスに従っているか？
- [ ] 学習レート・エポック数がメモリ制約下で最適か？
- [ ] 損失曲線が通常の学習曲線に従っているか？
- [ ] ファインチューニング後の推論品質が改善されたか？

#### ✅ Phase 2B: LCM 蒸留

**対象論文**: Luo et al. (2310.04378) "LCM: Latent Consistency Models for Fast Image Generation"

**論文から学ぶべき内容**:
- LCM の動作原理: 一貫性条件の学習によって ステップ数削減（20→4）
- 蒸留ロス関数: $\mathcal{L}_{LCM} = \ldots$（論文参照）
- スケジューラーの使い分け（DCScheduler vs LMScheduler）

**実装検証項目**:
- [ ] LCMScheduler の設定が論文の推奨値に従っているか？
- [ ] 蒸留ロスの計算が論文の式（Eq. 3等）に一致しているか？
- [ ] 4ステップ推論で 12倍高速化（3.8秒→0.3秒）を達成しているか？
- [ ] 品質低下が許容範囲内か？（主観評価 + LPIPS 測定）

**コード検証例**:
```python
# PHASE_2B_LCM_DISTILLATION.md での実装例
class LCMDistiller:
    """
    Luo et al. (2310.04378) に基づく LCM 蒸留
    
    論文の式: L_LCM = E[ || f_theta(x_t, t; c) - x_0 ||^2 ]
    
    検証ポイント:
    ✓ スケジューラー設定: num_inference_steps=4
    ✓ 蒸留ロス式が論文 Eq. 3 に一致
    ✓ 推論速度: 0.3-0.5 秒/画像（4ステップ）
    ✓ 品質評価: v1.5 との LPIPS 比較
    """
```

#### ✅ Phase 3: マルチモーダル操作

**対象論文**: 複数（ControlNet、Inpainting等）

**検証項目**:
- [ ] ControlNet の条件付き生成メカニズムが論文一致か？
- [ ] Inpainting の潜在空間操作が数学的に正確か？

#### ✅ Phase 4: デプロイ

**検証項目**:
- [ ] 本番環境での推論精度が開発環境と一致しているか？
- [ ] メモリ使用量が制約内か？

---

### 🔍 論文検証ワークフロー

各 Phase の実装後に、**必ず以下のワークフロー** を実行してください：

#### Step 1: 論文再読
```
実装完了 → 対応する論文をもう一度読む → 重要な式・アルゴリズムをノートに記録
```

#### Step 2: コード検証マッピング
```
論文の式  ← マッピング →  実装のコード
例: "蒸留ロス L_LCM = ..."  → def lcm_loss(predictions, targets):
```

#### Step 3: テスト実装
```python
# テストコードで検証
def test_lcm_consistency():
    """LCM の一貫性条件が満たされているか検証"""
    # 4ステップ推論で、潜在変数が一貫性条件 f(x_t, t) = x_0 に近い値になるか
    pass
```

#### Step 4: ベンチマーク実施
```
推論速度: 0.3秒/画像（論文が示す 12倍以上）
品質低下: LPIPS < 0.15（許容範囲）
```

#### Step 5: ドキュメント記述
```
実装ガイド (PHASE_*.md) に
- 論文のどの部分を実装したか
- コードがどこで論文の式に対応しているか
を必ず記述
```

---

## 🗂️ ファイル構成とルール

```
anime-character-generator/
│
├── 【v1.0版】
│   ├── character_generator_v1.py              ← ブログ対応版
│   └── anime_generator_colab_simple_v1.0.ipynb
│
├── 【v1.5版】
│   ├── character_generator_v1_lora.py         ← LoRA実装版
│   └── anime_generator_colab_lora_v1.5.ipynb  ⚠️ 試行版・警告付き
│
├── 【v2.0版】
│   ├── character_generator.py                 ← Phase 1-4の改善版
│   ├── IMPLEMENTATION_ROADMAP.md              ← 実装全体計画
│   ├── Amendment.md                          ← プロジェクト修正計画
│   │
│   └── 【実装ガイド】
│       ├── PHASE_1_PROMPT_OPTIMIZATION.md    ← LLM プロンプト設計
│       ├── dev_peft.md                       ← LoRA 学習実行
│       ├── PHASE_2B_LCM_DISTILLATION.md      ← 推論高速化
│       ├── PHASE_3_MULTIMODAL.md             ← マルチモーダル
│       └── PHASE_4_DEPLOYMENT.md             ← デプロイ
│
├── 【ブログ】
│   └── blog_articles/Day3-4_implementation_guide.md
│
└── 【設定ファイル】
    ├── .github/copilot-instructions.md       ← このファイル
    ├── requirements.txt
    └── .gitignore
```

---

## 📋 実装ガイドの構成

各 PHASE_*.md は以下の構成に従ってください：

```markdown
# Phase [N]: [目的]

## 📚 対応論文
- Ho et al. (2020) / Rombach et al. (2022) 等

## 🎯 実装目的
論文から何を学ぶか、何を実装するか

## 🔍 論文検証チェックリスト
- [ ] 論文の式[N]とコードの実装が一致しているか？
- [ ] 推奨パラメータが論文の実験一致か？
- [ ] ベンチマーク結果が論文の報告値と一致しているか？

## 💻 実装ガイド
ステップバイステップの実装手順

## ✅ 完了判定
- [ ] チェックポイント 1
- [ ] チェックポイント 2
```

---

## 🚀 GitHub Copilot への指示行為

### Copilot Chat で効果的な質問

#### ✅ 論文ベースの質問

```
Q: "Gao et al. (2306.13103) の多層冗長プロンプト設計を
   Anthropic Claude API で実装する際、どのように
   prompt_variants を生成すべき？
   論文の提案に従いながら実装したいです。"

期待される応答:
- 論文の該当個所（セクション、図表）の引用
- 設計パターンのコード例
- 検証方法
```

#### ✅ 実装検証の質問

```
Q: "このコードが Luo et al. (2310.04378) の LCM 蒸留ロス式
   (Eq. 3) に正確に対応しているか確認できますか？"

期待される応答:
- コードと論文式 の対応マッピング
- 計算が数学的に正確か
```

### ❌ 避けるべき質問

```
❌ "最速でコード書いて"          → 論文との対応が失われる
❌ "テンプレから生成して"        → 学術的背景が反映されない
❌ "とりあえず動く実装"          → 検証不可能
```

---

## 🔌 MCP (Model Context Protocol) 連携ガイド

このプロジェクトではMCPサーバーを活用して、**エラーが少ないコードを書き、修正が必要な箇所を最小限にする**ことに注力します。

### 🎯 MCP の役割分担

#### Serena — コード可読性と一貫性
- **用途**: プロジェクト全体のコードパターンを理解し、一貫した実装を確保
- **効果**: v1.0 → v1.5 → v2.0 の進化に沿ったコード構造を保証
- **結果**: リファクタリングや設計見直しが不要

#### Context7 — API互換性エラー防止
- **用途**: 最新の PyTorch, Diffusers, Transformers のドキュメントを参照
- **効果**: API変更やバージョン非互換性を実装前に検出
- **結果**: 「動かない」エラーが発生しない

#### playwright-mcp — 論文との整合性検証
- **用途**: 実装コードが論文の式・アルゴリズムと正確に対応しているかを確認
- **効果**: パラメータ値、計算順序、変数定義の正確性を証拠付きで検証
- **結果**: 修正学習・再実装が不要

### 📋 実装チェックリスト

**MCP活用による削減効果**

| 項目 | MCP未使用 | MCP使用 |
|------|---------|--------|
| 実装エラー発生率 | 30-50% | 0-5% |
| テンプレート見直し | 繁繁 | 不要 |
| 論文確認ラウンド | 3-5回 | 1回 |
| **修正が必要な箇所** | **多数** | **最小限** |

### 💡 使用例

```python
# ❌ 修正が多く発生するパターン（MCPなし）
# → API互換性エラー
# → v1.0との不整合
# → 論文の式と異なる実装

# ✅ MCPを活用したパターン
# 1. Serena でプロジェクト全体像を把握
#    → メソッド署名、クラス継承を統一
# 2. Context7 で最新API確認
#    → pipeline.to(device) の最適な形式を検証
# 3. playwright-mcp で論文確認
#    → Eq. 3 の計算式がコードと完全一致か確認
# 結果: エラーなし、修正不要で完成
```

