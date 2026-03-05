#!/usr/bin/env python3
"""
LLM ベースの堅牢なプロンプト生成エンジン v2

【Phase 2B + Phase 3統合版】
Gao et al. (2306.13103) + LCM-LoRA 論文ベース設計

対応パイプライン:
- ✅ Text-to-Image: 公式LCM-LoRA + anime LoRA
- ✅ ControlNet対応: スケッチ → 着彩生成
- ✅ プロンプト堅牢性: マルチレイヤーで摂動耐性強化

使用例:
    # Text-to-Image (LCM対応)
    generator = RobustPromptGenerator()
    result = generator.generate_prompt(
        "happy anime girl",
        use_lcm=True,
        controlnet_mode=False
    )
    print(result["positive_prompt"])
    print(result["lcm_settings"])  # {"guidance_scale": 1.5, "num_inference_steps": 4}
    
    # ControlNet統合 (スケッチ着彩)
    result = generator.generate_prompt(
        "anime girl, colorful outfit",
        use_lcm=True,
        controlnet_mode="lineart",
        controlnet_conditioning_scale=0.8
    )
"""

import os
import json
from typing import Dict, List, Optional
from pathlib import Path
from dotenv import load_dotenv


class RobustPromptGenerator:
    """
    LLM ベースの堅牢プロンプト生成（Google API / HuggingFace対応版）
    
    特徴:
    - Google Generative AI（Gemini）またはHuggingFaceローカルモデル選択可能
    - キャッシング機構（API コスト削減）
    - LCM-LoRA 対応設定の自動提案
    - ControlNet (lineart, scribble等) 対応
    - 信頼度スコア付き
    """
    
    def __init__(
        self,
        cache_dir: str = "./prompt_cache",
        use_google_api: bool = True
    ):
        """
        初期化
        
        Args:
            cache_dir: プロンプトキャッシュディレクトリ
            use_google_api: True=Google Generative AI, False=HuggingFace ローカル
        """
        load_dotenv()
        
        self.use_google_api = use_google_api
        self.client = None
        self.model_name = None
        
        if use_google_api:
            # Google Generative API
            api_key = os.getenv("Google_Api_Key")
            if not api_key:
                raise ValueError("❌ Google_Api_Key not set in .env")
            
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                self.client = genai.GenerativeModel("gemini-pro")
                self.model_name = "gemini-pro"
                print("✅ Using Google Generative AI (Gemini)")
            except ImportError:
                raise ImportError("❌ google-generativeai not installed. Run: pip install google-generativeai")
        else:
            # HuggingFace ローカルモデル
            model_name_env = os.getenv(
                "HUGGINGFACE_MODEL",
                "tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2"
            )
            
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch
                
                print(f"📦 Loading HuggingFace model: {model_name_env}")
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.dtype = torch.float16 if self.device == "cuda" else torch.float32
                
                self.tokenizer = AutoTokenizer.from_pretrained(model_name_env)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name_env,
                    torch_dtype=self.dtype,
                    device_map="auto" if self.device == "cuda" else None,
                    load_in_8bit=True if self.device == "cuda" else False
                )
                
                self.model_name = model_name_env
                print(f"✅ Using HuggingFace model ({self.device}): {model_name_env}")
                
            except ImportError as e:
                raise ImportError(f"❌ transformers または torch not installed: {e}")
            except Exception as e:
                raise RuntimeError(f"❌ Failed to load HuggingFace model: {e}")
        
        # キャッシュ初期化
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
            print(f"📚 Loaded {len(self.cache)} cached prompts")
    
    def _save_cache(self):
        """キャッシュをファイルに保存"""
        cache_file = self.cache_dir / "prompts.json"
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
    
    def generate_prompt(
        self,
        description: str,
        use_lcm: bool = True,
        controlnet_mode: Optional[str] = None,
        controlnet_conditioning_scale: float = 0.8,
        quality_level: str = "masterpiece"
    ) -> Dict:
        """
        マルチレイヤープロンプト生成（LCM + ControlNet 対応）
        
        Args:
            description: キャラクター描写（例: "happy anime girl, pink hair"）
            use_lcm: LCM-LoRA 用設定を含めるか
            controlnet_mode: ControlNet モード ("lineart", "scribble", None)
            controlnet_conditioning_scale: ControlNet の強度（0-1）
            quality_level: 品質レベル
        
        Returns:
            {
                "positive_prompt": str,
                "negative_prompt": str,
                "lcm_settings": {"guidance_scale": 1.5, "num_inference_steps": 4},
                "controlnet_settings": {...} or None,
                "confidence": float,
                "layers": dict,
                "reasoning": str
            }
        """
        
        # キャッシュキー生成
        cache_key = f"{description}_{use_lcm}_{controlnet_mode}"
        if cache_key in self.cache:
            print(f"📚 Cache hit: {cache_key}")
            return self.cache[cache_key]
        
        print(f"🤖 Generating prompt (LCM={use_lcm}, ControlNet={controlnet_mode})...")
        
        # プロンプト用のシステムメッセージ
        system_prompt = self._build_system_prompt(
            use_lcm=use_lcm,
            controlnet_mode=controlnet_mode
        )
        
        # ユーザーメッセージ
        user_message = self._build_user_message(
            description=description,
            use_lcm=use_lcm,
            controlnet_mode=controlnet_mode,
            quality_level=quality_level
        )
        
        try:
            if self.use_google_api:
                result = self._generate_with_google(system_prompt, user_message)
            else:
                result = self._generate_with_huggingface(system_prompt, user_message)
            
            # LCM設定を自動追加
            if use_lcm:
                result["lcm_settings"] = {
                    "guidance_scale": 1.5,
                    "num_inference_steps": 4,
                    "scheduler": "LCMScheduler",
                    "note": "Augmented PF-ODE により guidance=1.5 で w=7.5 相当効果"
                }
            
            # ControlNet設定を自動追加
            if controlnet_mode:
                result["controlnet_settings"] = {
                    "mode": controlnet_mode,
                    "conditioning_scale": controlnet_conditioning_scale,
                    "hint": "スケッチ/線画から構造を保持したまま着彩"
                }
            
            # キャッシュに保存
            self.cache[cache_key] = result
            self._save_cache()
            
            return result
        
        except json.JSONDecodeError:
            print("⚠️  JSON パースエラー。フォールバック使用")
            return self._fallback_prompt(description, use_lcm, controlnet_mode)
        
        except Exception as e:
            print(f"❌ Error: {e}")
            return self._fallback_prompt(description, use_lcm, controlnet_mode)
    
    def _generate_with_google(self, system_prompt: str, user_message: str) -> Dict:
        """Google Generative AI を使用して生成"""
        
        full_prompt = f"{system_prompt}\n\n{user_message}"
        
        response = self.client.generate_content(full_prompt)
        result_text = response.text
        
        # JSON 抽出（最初の { から最後の } まで）
        start_idx = result_text.find('{')
        end_idx = result_text.rfind('}') + 1
        
        if start_idx == -1 or end_idx == 0:
            raise json.JSONDecodeError("JSON not found", result_text, 0)
        
        json_str = result_text[start_idx:end_idx]
        result = json.loads(json_str)
        
        return result
    
    def _generate_with_huggingface(self, system_prompt: str, user_message: str) -> Dict:
        """HuggingFace ローカルモデルを使用して生成"""
        
        import torch
        
        full_prompt = f"{system_prompt}\n\n{user_message}"
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=1024,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        result_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # JSON 抽出
        start_idx = result_text.find('{')
        end_idx = result_text.rfind('}') + 1
        
        if start_idx == -1 or end_idx == 0:
            raise json.JSONDecodeError("JSON not found", result_text, 0)
        
        json_str = result_text[start_idx:end_idx]
        result = json.loads(json_str)
        
        return result
    
    def _build_system_prompt(
        self,
        use_lcm: bool,
        controlnet_mode: Optional[str]
    ) -> str:
        """システムプロンプト構築"""
        
        base_system = """あなたは Stable Diffusion v1.5 向けの高品質プロンプト生成エンジンです。

【基本ルール】
1. マルチレイヤー構造で堅牢なプロンプトを生成
2. 複数の同義表現を提供（Gao et al.の摂動耐性研究対応）
3. 感情タグは最低3つ、できれば5つ提供
4. スタイル指定は具体的で曖昧さ最小化
5. 負のプロンプトは必須（何を避けるか明確に）
6. 信頼度スコア（0-1.0）を返す
7. 応答は JSON 形式（UTF-8）"""
        
        if use_lcm:
            base_system += """

【LCM-LoRA統合（arXiv:2311.05556）】
- 1-4ステップで高速推論が必要
- guidance=1.5（Augmented PF-ODE で w=7.5 相当）
- プロンプトの正確性がより重要（少ないステップ=曖昧さ許容度低い）
- 強調タグは控えめに（()記号は1-2個程度）"""
        
        if controlnet_mode == "lineart":
            base_system += """

【ControlNet (Lineart) 統合】
- スケッチをベースに着彩・テクスチャ追加
- 線画の「構造」を尊重したプロンプト設計
- 例: "flowing hair lines, defined dress silhouette, clear facial features"
- 詳細すぎる記述は避ける（線画で既に構造が決定）"""
        
        elif controlnet_mode == "scribble":
            base_system += """

【ControlNet (Scribble) 統合】
- ラフスケッチをベースに精細化
- 「粗い線画」を想定してプロンプト設計
- 例: "sketch-like linework, defined composition, color harmony"
- 複雑さとシンプルさのバランス重視"""
        
        base_system += """

【出力 JSON 構造】
{
  "core": "1girl, anime character, master quality",
  "emotion_tags": ["happy", "cheerful", "smiling"],
  "style_descriptors": ["elegant", "detailed", "vibrant"],
  "quality_modifiers": ["masterpiece", "best quality", "highly detailed"],
  "negative_prompt": ["low quality", "blurry", "deformed"],
  "confidence": 0.85,
  "reasoning": "生成理由の説明"
}"""
        
        return base_system
    
    def _build_user_message(
        self,
        description: str,
        use_lcm: bool,
        controlnet_mode: Optional[str],
        quality_level: str
    ) -> str:
        """ユーザーメッセージ構築"""
        
        msg = f"""キャラクター描写: {description}
品質レベル: {quality_level}"""
        
        if use_lcm:
            msg += f"""

【LCM 推論設定】
- Guidance Scale: 1.5（Augmented PF-ODE で効果的）
- Inference Steps: 4
- Scheduler: LCMScheduler（ステップ削減対応）

このセッティングで高品質を得るには、プロンプトの明確性が必須です。"""
        
        if controlnet_mode:
            msg += f"""

【ControlNet: {controlnet_mode}】
基本となるスケッチから、以下を生成:
- 着彩（色選定）
- テクスチャ詳細化
- 光と影の強調

スケッチの線を尊重しながら、上記要素を加える必要があります。"""
        
        msg += """

上記を踏まえ、JSON形式で堅牢なプロンプトを生成してください。"""
        
        return msg
    
    def _fallback_prompt(
        self,
        description: str,
        use_lcm: bool,
        controlnet_mode: Optional[str]
    ) -> Dict:
        """API失敗時のフォールバック"""
        
        print("⚠️  Using fallback prompt generation")
        
        positive = f"1girl, {description}, masterpiece, best quality, highly detailed"
        negative = "low quality, blurry, deformed, ugly, bad anatomy"
        
        result = {
            "core": "1girl, anime character",
            "emotion_tags": [],
            "style_descriptors": [description],
            "quality_modifiers": ["masterpiece", "best quality"],
            "positive_prompt": positive,
            "negative_prompt": negative,
            "confidence": 0.5,
            "reasoning": "Fallback prompt (API failed)"
        }
        
        if use_lcm:
            result["lcm_settings"] = {
                "guidance_scale": 1.5,
                "num_inference_steps": 4,
                "scheduler": "LCMScheduler"
            }
        
        if controlnet_mode:
            result["controlnet_settings"] = {
                "mode": controlnet_mode,
                "conditioning_scale": 0.8
            }
        
        return result
    
    def batch_generate(
        self,
        descriptions: List[str],
        use_lcm: bool = True,
        controlnet_mode: Optional[str] = None
    ) -> List[Dict]:
        """
        複数プロンプトをバッチ生成
        
        Args:
            descriptions: キャラクター描写リスト
            use_lcm: LCM対応
            controlnet_mode: ControlNet モード
        
        Returns:
            プロンプト結果リスト
        """
        results = []
        for desc in descriptions:
            result = self.generate_prompt(
                desc,
                use_lcm=use_lcm,
                controlnet_mode=controlnet_mode
            )
            results.append(result)
        
        print(f"✅ Generated {len(results)} prompts")
        return results


def main():
    """デモ実行"""
    
    print("🚀 RobustPromptGenerator v2 Demo\n")
    
    try:
        # デフォルト: Google Generative AIを使用
        # ローカル実行する場合: use_google_api=False
        generator = RobustPromptGenerator(use_google_api=True)
        
        # テスト 1: LCM対応プロンプト
        print("="*60)
        print("Test 1: LCM-LoRA向けプロンプト生成")
        print("="*60)
        result1 = generator.generate_prompt(
            "happy anime girl with pink hair",
            use_lcm=True,
            controlnet_mode=None
        )
        print(f"\n✨ Positive: {result1['positive_prompt']}")
        print(f"❌ Negative: {result1['negative_prompt']}")
        print(f"⚙️  LCM Settings: {result1.get('lcm_settings')}")
        print(f"📊 Confidence: {result1['confidence']:.2f}\n")
        
        # テスト 2: ControlNet + LCM統合
        print("="*60)
        print("Test 2: ControlNet (Lineart) + LCM統合")
        print("="*60)
        result2 = generator.generate_prompt(
            "elegant anime girl, flowing dress",
            use_lcm=True,
            controlnet_mode="lineart",
            controlnet_conditioning_scale=0.8
        )
        print(f"\n✨ Positive: {result2['positive_prompt']}")
        print(f"⚙️  ControlNet: {result2.get('controlnet_settings')}")
        
        # テスト 3: バッチ生成
        print("\n" + "="*60)
        print("Test 3: バッチ生成（複数プロンプット）")
        print("="*60)
        descriptions = [
            "sad anime girl",
            "angry warrior character",
            "calm peaceful character"
        ]
        results = generator.batch_generate(descriptions, use_lcm=True)
        for i, r in enumerate(results, 1):
            print(f"{i}. {r['positive_prompt'][:60]}...")
        
        print("\n✅ All tests completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
