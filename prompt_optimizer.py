#!/usr/bin/env python3
"""
LLM ベースの堅牢なプロンプト生成エンジン

Gao et al. (2306.13103) による脆弱性研究に基づき、タイポ・グリフ攻撃などの
文字レベルのノイズに対する耐性を強化したプロンプト設計を実装。

背景：Gao et al. は、Text-to-Image モデルが「A photo of an astronaut」を
「A photo of an astornaut」（タイポ）に変えるだけで生成結果が劇的に変わる
ことを実験で証明した。このガイドでは、複数の類似トークンを使用することで
この脆弱性を軽減するアプローチを採用している。

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
from dotenv import load_dotenv


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
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY 環境変数が設定されていません")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache = {}
        self._load_cache()
        
        print("✅ RobustPromptGenerator initialized with Gemini API")
    
    def _load_cache(self):
        """前回のセッションからキャッシュをロード"""
        cache_file = self.cache_dir / "prompts.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
                print(f"✅ Loaded {len(self.cache)} cached prompts")
            except:
                print("⚠️  Failed to load cache, starting fresh")
                self.cache = {}
    
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
あなたは Stable Diffusion v1.5 向けの高品質プロンプト生成エンジンです。

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
        
        user_message = f"""{system_prompt}

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
            
            # JSON ブロックを抽出
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            
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
            
            print(f"✅ Generated prompt (confidence: {result['confidence']:.2f})")
            
            return result
        
        except json.JSONDecodeError as e:
            print(f"⚠️  Failed to parse JSON response: {e}")
            # フォールバック
            return self._fallback_prompt(emotion, style, quality_level)
        
        except Exception as e:
            print(f"❌ API Error: {e}")
            return self._fallback_prompt(emotion, style, quality_level)
    
    def validate_prompt(self, prompt: str) -> Dict:
        """
        プロンプト品質の検証
        
        Gao et al. (2306.13103) が示した脆弱性（タイポ・グリフ攻撃への敏感性）
        を考慮し、プロンプトの冗長性と堅牢性をスコア化。
        
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
        
        validation_prompt = f"""
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
        
        try:
            response = self.model.generate_content(
                contents=validation_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=300,
                    temperature=0.5
                )
            )
            
            response_text = response.text
            
            # JSON ブロックを抽出
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            
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
        
        try:
            response = self.model.generate_content(
                contents=compare_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=300,
                    temperature=0.5
                )
            )
            
            response_text = response.text
            
            # JSON ブロックを抽出
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            
            return json.loads(response_text)
        except:
            return {"error": "Comparison failed"}
    
    def _fallback_prompt(self, emotion: str, style: str, quality: str) -> Dict:
        """
        API 失敗時のフォールバック
        """
        print("⚠️  Using fallback prompt")
        
        emotion_map = {
            "happy": ["happy", "cheerful", "smiling", "bright"],
            "angry": ["angry", "fierce", "intense", "determined"],
            "sad": ["sad", "melancholic", "crying", "sorrowful"],
            "neutral": ["neutral", "calm", "peaceful", "serene"],
        }
        
        style_map = {
            "casual": ["casual clothes", "relaxed pose", "everyday outfit"],
            "formal": ["formal dress", "elegant", "sophisticated"],
            "magical": ["magical", "fantasy", "mystical", "enchanting"],
            "anime": ["anime style", "detailed", "expressive"],
        }
        
        emotion_tags = emotion_map.get(emotion, [emotion])[:3]
        style_desc = style_map.get(style, [style])[:3]
        
        positive = f"1girl, anime character, {', '.join(emotion_tags)}, {', '.join(style_desc)}, {quality}"
        negative = "low quality, blurry, deformed, ugly, bad anatomy"
        
        return {
            "positive_prompt": positive,
            "negative_prompt": negative,
            "confidence": 0.5,
            "layers": {
                "core": "1girl, anime character",
                "emotion_tags": emotion_tags,
                "style_descriptors": style_desc,
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
    
    print("🚀 Starting RobustPromptGenerator demo\n")
    
    try:
        generator = RobustPromptGenerator()
        
        # テスト 1: happy + formal
        print("\n" + "="*60)
        print("テスト 1: happy + formal")
        print("="*60)
        result1 = generator.generate_prompt("happy", "formal dress")
        print(f"\n✨ Positive Prompt:\n{result1['positive_prompt']}\n")
        print(f"❌ Negative Prompt:\n{result1['negative_prompt']}\n")
        print(f"📊 Confidence: {result1['confidence']:.2f}\n")
        
        # テスト 2: キャッシュ確認（同じプロンプットで再度実行）
        print("\n" + "="*60)
        print("テスト 2: キャッシュ確認（同じプロンプットで再度実行）")
        print("="*60)
        result2 = generator.generate_prompt("happy", "formal dress")
        print(f"✅ キャッシュから取得しました\n")
        
        # テスト 3: 別のプロンプット
        print("\n" + "="*60)
        print("テスト 3: sad + casual")
        print("="*60)
        result3 = generator.generate_prompt("sad", "casual")
        print(f"\n✨ Positive Prompt:\n{result3['positive_prompt']}\n")
        
        # テスト 4: プロンプット検証
        print("\n" + "="*60)
        print("テスト 4: プロンプット検証")
        print("="*60)
        validation = generator.validate_prompt(result1["positive_prompt"])
        if "error" not in validation:
            print(f"📊 Validation Result:\n{json.dumps(validation, ensure_ascii=False, indent=2)}\n")
        
        print("✅ デモ実行完了！")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
