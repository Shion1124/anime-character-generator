# Phase 1: LLM × プロンプト最適化実装ガイド

**対象フェーズ**: Phase 1 (LLM プロンプト最適化 × ロバストネス設計)  
**推定期間**: 1-2週間  
**基盤理論**: Gao et al. (2306.13103) 「Text-to-Image Robustness」  
**成果物**: RobustPromptGenerator クラス + 統合テスト + ブログ記事

---

## 📖 背景：なぜプロンプト最適化が必要か？

### 問題: Text-to-Image の脆弱性

Gao et al. (2306.13103) のセキュリティ研究が明らかにした脆弱性：

Text-to-Image 拡散モデルは、**極めて微細なテキストレベルの摂動に対して非常に脆弱**です。特に以下の点が実験で確認されました。

```
【最重要な発見】
課題 1: 文字レベルのノイズ（タイポ・グリフ攻撃）への脆弱性
   "A photo of an astronaut" → "A photo of an astornaut"（タイポ）
   → 生成画像のセマンティクス（意味内容）が劇的に変わる
   
   グリフ攻撃（視覚的に似た文字への置換）:「l」→「1」など
   → 同様に機能し、生成結果を大幅に変化させる

【追加の脆弱性】
課題 2: 類義語置換への敏感性
   "smile" vs "happy" vs "cheerful" → 微妙に異なる結果

課題 3: トークン間の相互干渉
   複数の修飾子を指定すると、一部が消失したり
   予期しない結果が生じる
```

### 解決策: マルチレイヤープロンプト設計（対策案）

Gao et al. が示した「**一文字のミスで結果が変わる**」という脆弱性を克服するため、本実装では**単一のトークンに頼らない冗長性を持たせたプロンプト設計**を採用します。

複数の類似トークンを並べることで、一部がノイズで失われても意図を維持するアンサンブル的アプローチ：

```
Layer 1: コア設定（基本要素、変更に強い）
         "1girl, anime character, detailed face"

Layer 2: 感情トークン（3-5 種類の表現）
         "happy", "cheerful", "smile", "bright expression", "joyful"

Layer 3: スタイル記述子（具体的、曖昧さ最小化）
         "watercolor", "soft shading", "pastel colors"

Layer 4: 品質修飾子（出力品質保証）
         "masterpiece", "best quality", "high detail"
```

---

## 🛠️ 実装ステップ

### Step 1: 環境構築

```bash
# Google Generative AI SDK のインストール
pip install google-generativeai
pip install python-dotenv  # 環境変数管理用

# 環境変数設定
export GEMINI_API_KEY="your-api-key-here"
# または .env ファイルに記載
```

**必要な API キー**:
- Google Gemini API キー（https://ai.google.dev/）

### Step 2: RobustPromptGenerator クラス実装

**ファイル**: `prompt_optimizer.py` を新規作成

```python
#!/usr/bin/env python3
"""
LLM ベースの堅牢なプロンプト生成エンジン

Gao et al. (2306.13103) に基づく、Text-to-Image Robustness 強化

使用例:
    generator = RobustPromptGenerator()
    result = generator.generate_prompt("happy", "formal dress")
    print(result["positive_prompt"])
"""

import google.generativeai as genai
import json
from typing import Dict, List, Optional
from pathlib import Path
import os


class RobustPromptGenerator:
    """
    LLM ベースの堅牢なプロンプト生成
    
    特徴:
    - Google Gemini API を使用した多層構造プロンプト生成
    - キャッシング機構（API コスト削減）
    - 信頼度スコア付き
    - バリデーション機能
    """
    
    def __init__(self, cache_dir: str = "./prompt_cache"):
        """
        初期化
        
        Args:
            cache_dir: キャッシュディレクトリ
        """
        # Gemini API 初期化
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY 環境変数が設定されていません")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache = {}
        self._load_cache()
    
    def _load_cache(self):
        """前回のセッションからキャッシュをロード"""
        cache_file = self.cache_dir / "prompts.json"
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                self.cache = json.load(f)
            print(f"✅ Loaded {len(self.cache)} cached prompts")
    
    def _save_cache(self):
        """キャッシュをファイルに保存"""
        cache_file = self.cache_dir / "prompts.json"
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
    
    def generate_prompt(
        self,
        emotion: str,
        style: str,
        quality_level: str = "masterpiece",
        additional_context: Optional[str] = None
    ) -> Dict:
        """
        マルチレイヤープロンプト生成
        
        Args:
            emotion: 感情（"happy", "angry", "sad" など）
            style: スタイル（"formal", "casual" など）
            quality_level: 品質レベル
            additional_context: 追加コンテキスト（オプション）
        
        Returns:
            {
                "positive_prompt": str,
                "negative_prompt": str,
                "confidence": float,
                "metadata": dict,
                "layers": dict
            }
        """
        
        # キャッシュ確認
        cache_key = f"{emotion}_{style}_{quality_level}"
        if cache_key in self.cache:
            print(f"📦 Using cached prompt for: {cache_key}")
            return self.cache[cache_key]
        
        print(f"🤖 Generating prompt: {emotion} + {style}...")
        
        # プロンプト生成用メッセージ
        system_prompt = """
你は Stable Diffusion v1.5 向けの高品質プロンプト生成エンジンです。

以下のルールに従ってください：
1. マルチレイヤー構造で プロンプトを生成
2. 複数の同義表現を提供（摂動耐性向上）
3. 感情タグは最低3つ、できれば5つ提供
4. スタイル指定は具体的で曖昧さ最小化
5. 負のプロンプトは必須（何を避けるか明確に）
6. 信頼度スコア（0-1.0）を返す

応答は JSON 形式で、以下の構造:
{
  "core": "基本ベース（変更に強い）",
  "emotion_tags": ["tag1", "tag2", "tag3", ...],
  "style_descriptors": ["style1", "style2", "style3", ...],
  "quality_modifiers": ["quality1", "quality2"],
  "negative_prompt": ["avoid1", "avoid2", ...],
  "confidence": 0.0-1.0,
  "reasoning": "生成理由の説明"
}
"""
        
        user_message = f"""
Stable Diffusion v1.5 向けプロンプト生成

感情: {emotion}
スタイル: {style}
品質: {quality_level}
{f'追加コンテキスト: {additional_context}' if additional_context else ''}

上記に基づいて、多層構造の堅牢なプロンプトを JSON 形式で生成してください。
"""
        
        try:
            response = self.model.generate_content(
                contents=user_message,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=400,
                    temperature=0.7
                )
            )
            
            # JSON 応答をパース
            response_text = response.text
            response_json = json.loads(response_text)
            
            # プロンプト合成
            positive_parts = [
                response_json.get("core", "1girl, anime character"),
                ", ".join(response_json.get("emotion_tags", [])),
                ", ".join(response_json.get("style_descriptors", [])),
                ", ".join(response_json.get("quality_modifiers", []))
            ]
            positive_prompt = ", ".join([p for p in positive_parts if p])
            
            negative_prompt = ", ".join(
                response_json.get("negative_prompt", ["low quality"])
            )
            
            result = {
                "positive_prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "confidence": response_json.get("confidence", 0.8),
                "layers": {
                    "core": response_json.get("core"),
                    "emotion_tags": response_json.get("emotion_tags", []),
                    "style_descriptors": response_json.get("style_descriptors", []),
                    "quality_modifiers": response_json.get("quality_modifiers", [])
                },
                "metadata": {
                    "emotion": emotion,
                    "style": style,
                    "quality_level": quality_level,
                    "reasoning": response_json.get("reasoning", "")
                }
            }
            
            # キャッシュに保存
            self.cache[cache_key] = result
            self._save_cache()
            
            return result
        
        except json.JSONDecodeError:
            print(f"⚠️  Failed to parse JSON response")
            # フォールバック
            return self._fallback_prompt(emotion, style, quality_level)
        
        except Exception as e:
            print(f"❌ API Error: {e}")
            return self._fallback_prompt(emotion, style, quality_level)
    
    def validate_prompt(self, prompt: str) -> Dict:
        """
        プロンプト品質の検証
        
        Gao et al. が示した脆弱性（タイポ・グリフ攻撃への敏感性）を考慮し、
        プロンプトの冗長性と堅牢性をスコア化。
        
        Args:
            prompt: 検証対象プロンプト
        
        Returns:
            {
                "overall_score": 0-10,
                "ambiguity": 0-10,
                "consistency": 0-10,
                "robustness": 0-10,
                "recommendations": [...]
            }
        """
        
        print(f"🔍 Validating prompt...")
        
        response = self.model.generate_content(
            contents={
                "role": "user",
                "content": f"""
Stable Diffusion プロンプトの品質を評価してください。

プロンプト: {prompt}

以下の観点で JSON 形式で評価結果を返してください:
{{
  "overall_score": 0-10,
  "ambiguity_score": 0-10,
  "consistency_score": 0-10,
  "robustness_score": 0-10,
  "issues": ["issue1", "issue2", ...],
  "recommendations": ["recommendation1", ...],
  "summary": "評価サマリー"
}}

スコアは高いほど良い品質を示します。
"""
            }]
        )
        
        try:
            response_text = message.content[0].text
            return json.loads(response_text)
        except:
            return {"error": "Validation failed"}
    
    def compare_prompts(self, prompt1: str, prompt2: str) -> Dict:
        """
        2つのプロンプトを比較
        
        Args:
            prompt1: プロンプト1
            prompt2: プロンプト2
        
        Returns:
            比較結果
        """
        
        print(f"⚖️  Comparing prompts...")
        
        compare_prompt = f"""
2つの Stable Diffusion プロンプトを比較してください。

プロンプト1: {prompt1}
プロンプト2: {prompt2}

JSON 形式で比較結果を返してください:
{{
  "better": 1 or 2,
  "reason": "どちらが優れているか、理由",
  "prompt1_score": 0-10,
  "prompt2_score": 0-10,
  "robustness_difference": "摂動耐性の差",
  "recommendations": [...]
}}
"""
        
        response = self.model.generate_content(
            contents=compare_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=300,
                temperature=0.5
            )
        )
        
        try:
            response_text = response.text
            return json.loads(response_text)
        except:
            return {"error": "Comparison failed"}
    
    def _fallback_prompt(self, emotion: str, style: str, quality: str) -> Dict:
        """
        API 失敗時のフォールバック
        """
        fallback_map = {
            "happy": ["happy", "cheerful", "smiling", "bright"],
            "angry": ["angry", "fierce", "intense", "determined"],
            "sad": ["sad", "melancholic", "tearful", "sorrowful"],
            "calm": ["calm", "peaceful", "serene", "tranquil"]
        }
        
        emotion_tags = fallback_map.get(emotion.lower(), ["neutral"])
        
        return {
            "positive_prompt": f"1girl, anime character, {', '.join(emotion_tags)}, {style}, {quality}",
            "negative_prompt": "low quality, worst quality, blurry",
            "confidence": 0.6,
            "layers": {
                "core": "1girl, anime character",
                "emotion_tags": emotion_tags,
                "style_descriptors": [style],
                "quality_modifiers": [quality]
            },
            "metadata": {
                "emotion": emotion,
                "style": style,
                "quality_level": quality,
                "reasoning": "Fallback prompt"
            }
        }


def main():
    """デモ実行"""
    
    generator = RobustPromptGenerator()
    
    # 例 1: 基本的なプロンプト生成
    print("\n" + "="*60)
    print("例 1: happy + formal dress")
    print("="*60)
    result = generator.generate_prompt("happy", "formal dress")
    print(f"✅ Positive: {result['positive_prompt']}")
    print(f"❌ Negative: {result['negative_prompt']}")
    print(f"📊 Confidence: {result['confidence']:.2%}")
    print(f"📋 Layers: {result['layers']}")
    
    # 例 2: プロンプト検証
    print("\n" + "="*60)
    print("例 2: プロンプト検証")
    print("="*60)
    validation = generator.validate_prompt(result['positive_prompt'])
    print(f"✅ Validation Result: {json.dumps(validation, indent=2)}")
    
    # 例 3: 複数パターン生成
    print("\n" + "="*60)
    print("例 3: 複数の感情・スタイル組み合わせ")
    print("="*60)
    patterns = [
        ("happy", "casual"),
        ("angry", "formal"),
        ("sad", "artistic")
    ]
    for emotion, style in patterns:
        result = generator.generate_prompt(emotion, style)
        print(f"\n{emotion} + {style}:")
        print(f"  → {result['positive_prompt'][:60]}...")


if __name__ == "__main__":
    main()
```

### Step 3: character_generator.py への統合

`character_generator.py` の `AnimeCharacterGenerator` クラスに `RobustPromptGenerator` を統合：

```python
# character_generator.py の先頭に追加
from prompt_optimizer import RobustPromptGenerator

class AnimeCharacterGenerator:
    def __init__(self, device: str = "auto", use_robust_prompts: bool = True):
        # ... 既存のコード ...
        
        # ロバストプロンプト生成エンジンの初期化（オプション）
        if use_robust_prompts:
            self.prompt_optimizer = RobustPromptGenerator()
            print("✅ Robust Prompt Generator enabled")
        else:
            self.prompt_optimizer = None
    
    def generate_image_with_optimized_prompt(
        self,
        emotion: str,
        style: str,
        quality_level: str = "masterpiece"
    ) -> Image:
        """
        ロバストプロンプトを使用した画像生成
        
        Args:
            emotion: 感情
            style: スタイル
            quality_level: 品質レベル
        
        Returns:
            生成画像
        """
        
        if not self.prompt_optimizer:
            raise ValueError("Robust Prompt Generator is not enabled")
        
        # LLM でプロンプト最適化
        prompt_data = self.prompt_optimizer.generate_prompt(
            emotion=emotion,
            style=style,
            quality_level=quality_level
        )
        
        print(f"📊 Confidence: {prompt_data['confidence']:.2%}")
        
        # 最適化されたプロンプトで画像生成
        image = self.pipe(
            prompt=prompt_data["positive_prompt"],
            negative_prompt=prompt_data["negative_prompt"],
            num_inference_steps=20,
            guidance_scale=7.5
        ).images[0]
        
        return image
```

### Step 4: テスト・評価

```python
# test_prompt_optimization.py

from prompt_optimizer import RobustPromptGenerator
from character_generator import AnimeCharacterGenerator
import json

def test_prompt_generation():
    """プロンプト生成のテスト"""
    
    generator = RobustPromptGenerator()
    
    test_cases = [
        ("happy", "formal"),
        ("sad", "casual"),
        ("angry", "artistic"),
        ("calm", "portrait")
    ]
    
    results = []
    
    for emotion, style in test_cases:
        result = generator.generate_prompt(emotion, style)
        results.append({
            "emotion": emotion,
            "style": style,
            "confidence": result["confidence"],
            "prompt": result["positive_prompt"][:80]
        })
    
    # 結果保存
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("✅ Test results saved to test_results.json")

def test_image_generation():
    """画像生成のテスト"""
    
    generator = AnimeCharacterGenerator(use_robust_prompts=True)
    
    # ロバストプロンプトで生成
    image = generator.generate_image_with_optimized_prompt(
        emotion="happy",
        style="formal"
    )
    
    image.save("test_output_robust.png")
    print("✅ Image saved to test_output_robust.png")

if __name__ == "__main__":
    print("Running tests...")
    test_prompt_generation()
    test_image_generation()
```

---

## 📊 期待される改善効果

### 定性的改善

| 指標 | Before | After |
|------|--------|-------|
| プロンプト多様性 | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| 攻撃耐性 | N/A | ⭐⭐⭐⭐ |
| キャラ一貫性 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 生成品質 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### 定量的改善

```
指標: 同じ感情・スタイルの異なる実行結果の一貫性

Before (固定プロンプト):
  実行1: 生成画像 A
  実行2: 生成画像 B（Aと大幅に異なる）
  一貫性スコア: 0.6

After (マルチレイヤープロンプト):
  実行1: 生成画像 A'
  実行2: 生成画像 A'' (A' とよく似ている)
  一貫性スコア: 0.85+
```

---

## 🧪 テスト・デバッグ

### 環境変数確認

```bash
# .env ファイルを作成
echo "GEMINI_API_KEY=your-key-here" > .env

# Python では以下で読み込み
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
```

### よくあるエラーと対処

#### エラー 1: `google.api_core.exceptions.AuthenticationError`
```
原因: API キーが不正
対処: export GEMINI_API_KEY="..." を正しく設定
```

#### エラー 2: `json.JSONDecodeError`
```
原因: Gemini の応答が JSON でない
対処: システムプロンプトを確認、モデル更新
```

#### エラー 3: `Rate limit exceeded`
```
原因: API リクエスト過多
対処: キャッシング機構が有効か確認、キャッシュ `prompt_cache/prompts.json` を確認
```

---

## 📝 ブログ記事執筆

**テーマ**: 「LLM × 論文ベースのプロンプト設計」

**構成**:
1. Text-to-Image ロバストネスの解説（Gao et al. 2306.13103）
2. マルチレイヤープロンプトの仕組み
3. 実装デモ（コード例）
4. 一貫性の改善効果
5. 今後の展開（Phase 2 へ）

---

## ✅ 完了チェックリスト

- [ ] Google Generative AI SDK セットアップ
- [ ] `prompt_optimizer.py` 実装完了
- [ ] テスト・デバッグ完了
- [ ] `character_generator.py` 統合完了
- [ ] ローカルで `python test_prompt_optimization.py` で動作確認
- [ ] ブログ記事執筆・公開
- [ ] 本番環境での動作確認

---

**次のステップ**: Phase 1 完了後、dev_peft.md に従って **Phase 2A: Colab LoRA 学習** へ

