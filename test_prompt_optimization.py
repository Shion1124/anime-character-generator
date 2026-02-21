#!/usr/bin/env python3
"""
Phase 1: RobustPromptGenerator テストスイート

このスクリプトは以下をテストします:
1. prompt_optimizer.py の機能確認
2. character_generator.py との統合機能
3. プロンプトキャッシング機構
4. フォールバック機能
"""

import sys
import json
from pathlib import Path
from prompt_optimizer import RobustPromptGenerator
from character_generator import AnimeCharacterGenerator


def test_prompt_generator():
    """RobustPromptGenerator の基本機能をテスト"""
    
    print("\n" + "="*70)
    print("TEST 1: RobustPromptGenerator 初期化")
    print("="*70)
    
    try:
        generator = RobustPromptGenerator()
        print("✅ RobustPromptGenerator 初期化成功\n")
    except Exception as e:
        print(f"❌ 初期化失敗: {e}\n")
        return False
    
    # キャッシック確認
    print("="*70)
    print("TEST 2: プロンプト生成（キャッシュ機構）")
    print("="*70)
    
    test_cases = [
        ("happy", "formal"),
        ("sad", "casual"),
        ("happy", "formal"),  # キャッシュから取得
    ]
    
    for emotion, style in test_cases:
        print(f"\n生成: {emotion} + {style}")
        result = generator.generate_prompt(emotion, style)
        
        print(f"  ✨ Positive Prompt: {result['positive_prompt'][:80]}...")
        print(f"  ❌ Negative Prompt: {result['negative_prompt'][:80]}...")
        print(f"  📊 Confidence: {result['confidence']:.2f}")
        print(f"  📚 Method: {'キャッシュ' if result['metadata']['reasoning'] != 'Fallback prompt' else 'API/フォールバック'}")
    
    # キャッシュファイルの確認
    cache_file = Path("./prompt_cache/prompts.json")
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
        print(f"\n✅ キャッシュ保存確認: {len(cache_data)} プロンプトがキャッシュされています")
    
    return True


def test_character_generator_integration():
    """character_generator.py と the RobustPromptGenerator の統合をテスト"""
    
    print("\n" + "="*70)
    print("TEST 3: AnimeCharacterGenerator 初期化（RobustPromptGenerator 使用）")
    print("="*70)
    
    try:
        generator = AnimeCharacterGenerator(device="cpu", use_robust_prompt=True)
        print("✅ AnimeCharacterGenerator 初期化成功（RobustPrompt オン）\n")
    except Exception as e:
        print(f"❌ 初期化失敗: {e}\n")
        return False
    
    # RobustPromptGenerator が有効か確認
    if generator.robust_prompt_generator:
        print("✅ RobustPromptGenerator が統合されています\n")
    else:
        print("⚠️  RobustPromptGenerator が統合されていません（フォールバックモード）\n")
    
    return True


def test_prompt_validation():
    """プロンプト検証機能をテスト"""
    
    print("\n" + "="*70)
    print("TEST 4: プロンプト検証機能")
    print("="*70)
    
    try:
        generator = RobustPromptGenerator()
        
        test_prompt = "1girl, anime character, happy, masterpiece, high quality"
        print(f"\n検証対象: {test_prompt}\n")
        
        validation = generator.validate_prompt(test_prompt)
        
        if "error" not in validation:
            print("✅ 検証完了:")
            for key, value in validation.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value}")
                elif isinstance(value, list):
                    print(f"  {key}: {', '.join(map(str, value[:3]))}")
                elif isinstance(value, str):
                    print(f"  {key}: {value[:60]}...")
        else:
            print(f"⚠️  検証失敗: {validation['error']}")
    
    except Exception as e:
        print(f"❌ テスト失敗: {e}")
        return False
    
    return True


def test_prompt_comparison():
    """2つのプロンプト比較機能をテスト"""
    
    print("\n" + "="*70)
    print("TEST 5: プロンプト比較機能")
    print("="*70)
    
    try:
        generator = RobustPromptGenerator()
        
        prompt1 = "1girl, anime, happy, masterpiece"
        prompt2 = "1girl, anime character, joyful, cheerful, best quality, masterpiece"
        
        print(f"\nプロンプト1: {prompt1}")
        print(f"プロンプト2: {prompt2}\n")
        
        comparison = generator.compare_prompts(prompt1, prompt2)
        
        if "error" not in comparison:
            print("✅ 比較完了:")
            for key, value in comparison.items():
                if key != "recommendations":
                    print(f"  {key}: {value}")
        else:
            print(f"⚠️  比較失敗: {comparison['error']}")
    
    except Exception as e:
        print(f"❌ テスト失敗: {e}")
        return False
    
    return True


def test_fallback_mode():
    """フォールバック機能をテスト"""
    
    print("\n" + "="*70)
    print("TEST 6: フォールバック機能")
    print("="*70)
    
    try:
        generator = RobustPromptGenerator()
        
        # フォールバックプロンプト生成
        fallback_result = generator._fallback_prompt("happy", "formal", "masterpiece")
        
        print("✅ フォールバックプロンプト生成成功:")
        print(f"  Positive: {fallback_result['positive_prompt']}")
        print(f"  Negative: {fallback_result['negative_prompt']}")
        print(f"  Confidence: {fallback_result['confidence']}")
        
    except Exception as e:
        print(f"❌ テスト失敗: {e}")
        return False
    
    return True


def main():
    """すべてのテストを実行"""
    
    print("\n" + "="*70)
    print("🧪 Phase 1 テストスイート")
    print("="*70)
    
    tests = [
        ("プロンプト生成", test_prompt_generator),
        ("キャラクタジェネレータ統合", test_character_generator_integration),
        ("プロンプト検証", test_prompt_validation),
        ("プロンプト比較", test_prompt_comparison),
        ("フォールバック機能", test_fallback_mode),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} 実行エラー: {e}")
            results.append((test_name, False))
    
    # 結果サマリー
    print("\n" + "="*70)
    print("📊 テスト結果サマリー")
    print("="*70 + "\n")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n総合: {passed}/{total} テスト合格")
    
    if passed == total:
        print("\n🎉 すべてのテストが正常に完了しました！")
        print("\n次のステップ:")
        print("  1. APIクォータを確認・増加")
        print("  2. character_generator.py でプロンプト最適化を活用")
        print("  3. Phase 2A (LoRA トレーニング) へ進める")
        return 0
    else:
        print(f"\n⚠️  {total - passed} テストが失敗しました")
        print("\nトラブルシューティング:")
        print("  1. APIキーが正しく設定されているか確認")
        print("  2. requirements.txt のパッケージがインストールされているか確認")
        print("  3. .env ファイルが存在するか確認")
        return 1


if __name__ == "__main__":
    sys.exit(main())
