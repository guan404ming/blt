"""
Constraint Validator
驗證翻譯結果是否滿足音樂約束
"""

from .models import LyricTranslation, MusicConstraints, ValidationResult
from .feature_extractor import FeatureExtractor


class ConstraintValidator:
    """約束驗證器"""

    def __init__(self, target_lang: str = "Chinese"):
        self.target_lang = target_lang
        self.extractor = FeatureExtractor(target_lang=target_lang)

    def validate(
        self, translation: LyricTranslation, constraints: MusicConstraints
    ) -> ValidationResult:
        """
        驗證翻譯是否滿足約束

        Args:
            translation: 翻譯結果
            constraints: 目標約束

        Returns:
            ValidationResult: 驗證結果
        """
        errors = []
        scores = {}  # 改用字典來儲存各項分數及權重

        # 1. 驗證音節數（最高優先級）
        length_score, length_errors = self._validate_length(
            translation.syllable_counts, constraints.syllable_counts
        )
        scores["syllable"] = {"score": length_score, "weight": 0.7}  # 70% 權重
        errors.extend(length_errors)

        # 2. 驗證押韻（中等優先級，可以放寬）
        if constraints.rhyme_scheme:
            rhyme_score, rhyme_errors = self._validate_rhyme(
                translation.rhyme_endings, constraints.rhyme_scheme
            )
            scores["rhyme"] = {"score": rhyme_score, "weight": 0.2}  # 20% 權重
            # 只有在押韻完全不符合時才加入錯誤（放寬要求）
            if rhyme_score < 0.5:  # 只有低於 50% 才算錯誤
                errors.extend(rhyme_errors)

        # 3. 驗證詞邊界（最低優先級）
        if constraints.pause_positions:
            boundary_score, boundary_errors = self._validate_boundaries(
                translation.translated_lines, constraints.pause_positions
            )
            scores["boundary"] = {"score": boundary_score, "weight": 0.1}  # 10% 權重
            # 詞邊界錯誤不計入總錯誤（僅作參考）

        # 計算加權平均分數
        total_weight = sum(item["weight"] for item in scores.values())
        overall_score = (
            sum(item["score"] * item["weight"] for item in scores.values())
            / total_weight
            if total_weight > 0
            else 0.0
        )

        # 音節數必須完全正確才算通過（其他約束可以有誤差）
        syllable_perfect = length_score == 1.0
        passed = syllable_perfect and len(errors) == 0
        feedback = self._generate_feedback(passed, errors, overall_score)

        return ValidationResult(
            passed=passed,
            errors=errors,
            score=overall_score,
            feedback=feedback,
        )

    def _validate_length(
        self, actual_counts: list[int], target_counts: list[int]
    ) -> tuple[float, list[dict]]:
        """驗證音節數"""
        errors = []

        if len(actual_counts) != len(target_counts):
            errors.append(
                {
                    "type": "length",
                    "message": f"行數不匹配: 期望 {len(target_counts)} 行，實際 {len(actual_counts)} 行",
                }
            )
            return 0.0, errors

        mismatches = []
        for i, (actual, target) in enumerate(zip(actual_counts, target_counts)):
            if actual != target:
                mismatches.append(i)
                errors.append(
                    {
                        "type": "length",
                        "line": i + 1,
                        "expected": target,
                        "actual": actual,
                        "message": f"第 {i + 1} 行音節數不符: 期望 {target}，實際 {actual}",
                    }
                )

        # 計算準確率
        accuracy = (len(target_counts) - len(mismatches)) / len(target_counts)
        return accuracy, errors

    def _validate_rhyme(
        self, rhyme_endings: list[str], rhyme_scheme: str
    ) -> tuple[float, list[dict]]:
        """驗證押韻"""
        errors = []

        if len(rhyme_endings) != len(rhyme_scheme):
            errors.append(
                {
                    "type": "rhyme",
                    "message": f"押韻方案長度不匹配: 期望 {len(rhyme_scheme)}，實際 {len(rhyme_endings)}",
                }
            )
            return 0.0, errors

        # 構建期望的押韻組
        rhyme_groups = {}
        for i, label in enumerate(rhyme_scheme):
            if label not in rhyme_groups:
                rhyme_groups[label] = []
            rhyme_groups[label].append(i)

        # 檢查同組的韻腳是否相同
        mismatches = 0
        for label, indices in rhyme_groups.items():
            if len(indices) < 2:
                continue  # 單獨的韻腳不需要檢查

            # 取第一個作為基準
            base_rhyme = rhyme_endings[indices[0]]

            for idx in indices[1:]:
                actual_rhyme = rhyme_endings[idx]
                if not self._rhymes_with(base_rhyme, actual_rhyme):
                    mismatches += 1
                    errors.append(
                        {
                            "type": "rhyme",
                            "lines": [indices[0] + 1, idx + 1],
                            "expected_rhyme": base_rhyme,
                            "actual_rhyme": actual_rhyme,
                            "message": f"第 {idx + 1} 行與第 {indices[0] + 1} 行應押韻但不押韻",
                        }
                    )

        # 計算準確率
        total_checks = sum(
            len(indices) - 1 for indices in rhyme_groups.values() if len(indices) > 1
        )
        accuracy = (
            (total_checks - mismatches) / total_checks if total_checks > 0 else 1.0
        )

        return accuracy, errors

    def _rhymes_with(self, rhyme1: str, rhyme2: str) -> bool:
        """判斷兩個韻腳是否押韻"""
        if not rhyme1 or not rhyme2:
            return False

        # 簡單判斷: 韻腳相同或包含關係
        return rhyme1 == rhyme2 or rhyme1 in rhyme2 or rhyme2 in rhyme1

    def _validate_boundaries(
        self, translated_lines: list[str], pause_positions: list[int]
    ) -> tuple[float, list[dict]]:
        """驗證詞邊界"""
        errors = []

        # TODO: 實作詞邊界驗證
        # 需要分詞並檢查詞邊界是否在指定位置

        # 暫時返回完美分數
        return 1.0, errors

    def _generate_feedback(self, passed, errors, score) -> str:
        """根據驗證結果生成反饋文本"""
        if passed:
            return "✓ 所有約束都已滿足"

        feedback_parts = ["翻譯存在以下問題需要修正:\n"]

        # 統計各類錯誤數量
        length_errors = [e for e in errors if e["type"] == "length"]
        rhyme_errors = [e for e in errors if e["type"] == "rhyme"]
        boundary_errors = [e for e in errors if e["type"] == "boundary"]

        # 添加統計摘要
        if length_errors:
            feedback_parts.append(
                f"\n【音節數約束】共有 {len(length_errors)} 行不滿足:"
            )
            # 按行號排序
            length_errors_sorted = sorted(length_errors, key=lambda e: e.get("line", 0))
            for error in length_errors_sorted:
                line_num = error.get("line", "?")
                actual = error.get("actual", "?")
                expected = error.get("expected", "?")
                diff = (
                    actual - expected
                    if isinstance(actual, int) and isinstance(expected, int)
                    else "?"
                )
                diff_str = f"({diff:+d})" if isinstance(diff, int) else ""
                feedback_parts.append(
                    f"  第 {line_num} 行: 本次 {actual} 音節 → 目標 {expected} 音節 {diff_str * -1}"
                )

        if rhyme_errors:
            feedback_parts.append(f"\n【押韻約束】共有 {len(rhyme_errors)} 處不滿足:")
            for error in rhyme_errors:
                feedback_parts.append(f"  - {error['message']}")

        if boundary_errors:
            feedback_parts.append(
                f"\n【詞邊界約束】共有 {len(boundary_errors)} 處不滿足:"
            )
            for error in boundary_errors:
                feedback_parts.append(f"  - {error['message']}")

        # 添加整體評分
        feedback_parts.append(f"\n【整體評分】{score:.2%}")

        return "\n".join(feedback_parts)
