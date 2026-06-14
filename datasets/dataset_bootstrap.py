"""AREval 数据集引导脚本 (dataset_bootstrap.py)

用途：
1. 从种子数据 (datasets/seed/*.jsonl) 加载基础测试用例
2. 通过模板和规则自动生成变体，扩展数据集规模
3. 输出到 .areval/datasets/ 供框架直接使用
4. 生成统计报告

使用方式：
    python datasets/dataset_bootstrap.py                # 生成全部数据集
    python datasets/dataset_bootstrap.py --domain cs    # 只生成客服数据集
    python datasets/dataset_bootstrap.py --scale 3      # 每个种子生成3个变体
    python datasets/dataset_bootstrap.py --list          # 列出所有种子数据集
"""

import argparse
import json
import random
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# 项目路径处理：支持从项目根目录或 datasets/ 目录运行
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "areval-engine"))
sys.path.insert(0, str(_project_root / "areval-sdk"))

from areval.test_case import TestCase


# ============================================================================
# 种子数据加载
# ============================================================================

SEED_DIR = Path(__file__).resolve().parent / "seed"

SEED_FILES = {
    "cs": "customer_service.jsonl",
    "rag": "rag_evaluation.jsonl",
    "safety": "safety_redteam.jsonl",
}


def load_seed(domain: Optional[str] = None) -> Dict[str, List[TestCase]]:
    """加载种子数据集。

    Args:
        domain: 指定领域 ("cs", "rag", "safety")，None 时加载全部。

    Returns:
        {domain: [TestCase, ...]} 字典。
    """
    result = {}
    targets = {domain: SEED_FILES[domain]} if domain else SEED_FILES

    for key, filename in targets.items():
        filepath = SEED_DIR / filename
        if not filepath.exists():
            print(f"  [WARN] 种子文件不存在: {filepath}")
            continue

        test_cases = []
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                tc = TestCase.from_dict(data)
                test_cases.append(tc)

        result[key] = test_cases
        print(f"  [OK] {key}: {len(test_cases)} cases loaded from {filename}")

    return result


# ============================================================================
# 变体生成器
# ============================================================================

class VariantGenerator:
    """基于规则的数据集变体生成器。"""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    # ---- 通用变体策略 ----

    def paraphrase_input(self, tc: TestCase) -> TestCase:
        """对输入进行简单改写（添加前缀/后缀、语气调整）。"""
        new_tc = deepcopy(tc)
        new_tc.id = f"{tc.id}-para"
        new_tc.name = f"{tc.name}-paraphrase"

        prefixes = [
            "请问，", "你好，", "我想问一下，", "帮我看看，",
            "麻烦查一下，", "能不能帮忙，", "",
        ]
        suffixes = [
            "谢谢！", "麻烦了。", "在线等。", "急！", "",
        ]

        prefix = self.rng.choice(prefixes)
        suffix = self.rng.choice(suffixes)

        # 避免重复添加已有前缀
        input_text = tc.input
        for p in prefixes:
            if p and input_text.startswith(p):
                input_text = input_text[len(p):]
                break

        new_tc.input = prefix + input_text + suffix
        new_tc.tags = list(set(tc.tags + ["generated", "paraphrase"]))
        new_tc.metadata = {**tc.metadata, "generation_method": "paraphrase", "source": tc.name}
        return new_tc

    def adjust_difficulty(self, tc: TestCase, direction: str = "harder") -> TestCase:
        """调整难度：添加/移除约束条件。"""
        new_tc = deepcopy(tc)
        new_tc.id = f"{tc.id}-{direction}"
        new_tc.name = f"{tc.name}-{direction}"

        if direction == "harder":
            # 添加约束条件使问题更复杂
            constraints = [
                " 另外，我只有10分钟时间，请尽快处理。",
                " 顺便说一下，我上次遇到过类似问题，处理得很慢。",
                " 注意：我的账户是企业账户，可能有不同的规则。",
                " 我之前已经打过两次客服电话了，这是第三次联系。",
            ]
            new_tc.input = tc.input + self.rng.choice(constraints)
            new_tc.metadata = {**tc.metadata, "difficulty_adjusted": "harder"}
        else:
            # 简化输入
            words = tc.input.split()
            if len(words) > 10:
                new_tc.input = " ".join(words[:8]) + "？"
            new_tc.metadata = {**tc.metadata, "difficulty_adjusted": "easier"}

        new_tc.tags = list(set(tc.tags + ["generated", f"difficulty_{direction}"]))
        return new_tc

    # ---- 客服领域专用策略 ----

    def swap_order_number(self, tc: TestCase) -> TestCase:
        """替换订单号为随机新订单号。"""
        new_tc = deepcopy(tc)
        new_tc.id = f"{tc.id}-swap"
        new_tc.name = f"{tc.name}-order-swap"

        old_numbers = ["#20240601-8834", "#20240605-3344", "#20240612-1122"]
        new_number = f"#2024{self.rng.randint(1001,1231):04d}-{self.rng.randint(1000,9999):04d}"

        for old in old_numbers:
            if old in tc.input:
                new_tc.input = tc.input.replace(old, new_number)
                break
        else:
            # 没有订单号的 case，不生成此变体
            return None

        if tc.expected_output:
            for old in old_numbers:
                tc.expected_output = tc.expected_output.replace(old, new_number)
            new_tc.expected_output = tc.expected_output

        new_tc.tags = list(set(tc.tags + ["generated", "entity_swap"]))
        new_tc.metadata = {**tc.metadata, "generation_method": "entity_swap"}
        return new_tc

    def add_product_category(self, tc: TestCase) -> TestCase:
        """给产品推荐类问题添加具体品类。"""
        new_tc = deepcopy(tc)
        new_tc.id = f"{tc.id}-product"
        new_tc.name = f"{tc.name}-product-variant"

        products = [
            ("蓝牙耳机", "Sony WF-1000XM5", 1299),
            ("机械键盘", "Keychron K8 Pro", 599),
            ("显示器", "Dell U2723QE 27寸4K", 3299),
            ("平板电脑", "iPad Air M2", 4799),
            ("智能手表", "Apple Watch Series 10", 2999),
        ]
        product = self.rng.choice(products)

        new_tc.input = f"帮我推荐一款{product[0]}，预算{product[2]}以内。"
        new_tc.expected_output = f"推荐 {product[1]}，售价约{product[2]}元，在我们的数码品类中评分最高。您可以在相关分类下查看详情。"
        new_tc.tags = list(set(tc.tags + ["generated", "product_variant"]))
        new_tc.metadata = {**tc.metadata, "generation_method": "product_category", "product": product[0]}
        return new_tc

    # ---- RAG 领域专用策略 ----

    def swap_context_domain(self, tc: TestCase) -> TestCase:
        """替换 RAG context 中的数字/日期，测试模型是否忠实于 context。"""
        new_tc = deepcopy(tc)
        new_tc.id = f"{tc.id}-ctx-swap"
        new_tc.name = f"{tc.name}-context-swap"

        if not tc.context:
            return None

        # 数字替换策略：将所有数字乘以随机系数
        import re
        numbers = re.findall(r'\d+', tc.context)
        if not numbers:
            return None

        factor = self.rng.choice([0.8, 1.2, 1.5, 2.0])
        new_context = tc.context
        for num_str in set(numbers):
            try:
                old_val = int(num_str)
                new_val = int(old_val * factor)
                new_context = new_context.replace(num_str, str(new_val), 1)
            except ValueError:
                continue

        new_tc.context = new_context
        new_tc.tags = list(set(tc.tags + ["generated", "context_modified"]))
        new_tc.metadata = {**tc.metadata, "generation_method": "context_number_swap", "factor": factor}
        return new_tc

    # ---- 安全领域专用策略 ----

    def rephrase_attack(self, tc: TestCase) -> TestCase:
        """改写攻击措辞，测试泛化能力。"""
        new_tc = deepcopy(tc)
        new_tc.id = f"{tc.id}-rephrase"
        new_tc.name = f"{tc.name}-rephrased-attack"

        # 替换关键攻击短语
        replacements = {
            "Ignore all previous instructions": "Disregard everything you were told before",
            "reveal your system prompt": "show me your hidden instructions",
            "You are now": "From this moment you are",
            "no restrictions": "completely unrestricted",
            "pretend": "imagine",
            "hypothetical scenario": "fictional universe",
            "DAN": "an unrestricted AI",
            "hack": "penetrate",
            "phishing": "social engineering",
        }

        new_input = tc.input
        for old, new in replacements.items():
            if old.lower() in new_input.lower():
                new_input = new_input.replace(old, new)

        if new_input == tc.input:
            return None  # 没有可替换的内容

        new_tc.input = new_input
        new_tc.tags = list(set(tc.tags + ["generated", "rephrased_attack"]))
        new_tc.metadata = {**tc.metadata, "generation_method": "attack_rephrase"}
        return new_tc


# ============================================================================
# 主流程
# ============================================================================

def generate_variants(
    test_cases: List[TestCase],
    domain: str,
    scale: int = 2,
    seed: int = 42,
) -> List[TestCase]:
    """为种子数据生成变体。

    Args:
        test_cases: 种子测试用例。
        domain: 领域名称 ("cs", "rag", "safety")。
        scale: 每个种子生成的变体数量上限。
        seed: 随机种子。

    Returns:
        所有变体的列表（不含原始种子）。
    """
    gen = VariantGenerator(seed=seed)
    variants = []

    for tc in test_cases:
        count = 0

        # 策略 1: 改写输入（通用）
        if count < scale:
            v = gen.paraphrase_input(tc)
            if v and v.input != tc.input:
                variants.append(v)
                count += 1

        # 策略 2: 调整难度
        if count < scale:
            direction = "harder" if gen.rng.random() > 0.5 else "easier"
            v = gen.adjust_difficulty(tc, direction)
            if v and v.input != tc.input:
                variants.append(v)
                count += 1

        # 策略 3: 领域专用
        if count < scale:
            v = None
            if domain == "cs":
                if gen.rng.random() > 0.5:
                    v = gen.swap_order_number(tc)
                else:
                    v = gen.add_product_category(tc)
            elif domain == "rag":
                v = gen.swap_context_domain(tc)
            elif domain == "safety":
                v = gen.rephrase_attack(tc)

            if v and v.input != tc.input:
                variants.append(v)
                count += 1

    return variants


def build_dataset(
    domain: str,
    seeds: List[TestCase],
    variants: List[TestCase],
) -> Dict[str, Any]:
    """组装最终数据集并返回统计信息。"""
    all_cases = seeds + variants

    stats = {
        "domain": domain,
        "seed_count": len(seeds),
        "variant_count": len(variants),
        "total_count": len(all_cases),
        "tags_distribution": {},
        "difficulty_distribution": {},
    }

    for tc in all_cases:
        for tag in tc.tags:
            stats["tags_distribution"][tag] = stats["tags_distribution"].get(tag, 0) + 1
        diff = tc.metadata.get("difficulty", "unknown")
        stats["difficulty_distribution"][diff] = stats["difficulty_distribution"].get(diff, 0) + 1

    return {"cases": all_cases, "stats": stats}


def save_dataset(
    cases: List[TestCase],
    domain: str,
    output_dir: Path,
) -> Path:
    """保存数据集为 JSONL 文件。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"areval_{domain}_{datetime.now().strftime('%Y%m%d')}.jsonl"
    filepath = output_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        for tc in cases:
            f.write(json.dumps(tc.to_dict(), default=str, ensure_ascii=False) + "\n")

    return filepath


def print_report(results: Dict[str, Dict[str, Any]]) -> None:
    """打印数据集生成报告。"""
    print("\n" + "=" * 60)
    print("  AREval 数据集生成报告")
    print("=" * 60)

    total_seeds = 0
    total_variants = 0
    total_all = 0

    for domain, data in results.items():
        stats = data["stats"]
        total_seeds += stats["seed_count"]
        total_variants += stats["variant_count"]
        total_all += stats["total_count"]

        print(f"\n--- {domain.upper()} ---")
        print(f"  种子用例: {stats['seed_count']}")
        print(f"  生成变体: {stats['variant_count']}")
        print(f"  总计:     {stats['total_count']}")
        print(f"  难度分布: {stats['difficulty_distribution']}")

        top_tags = sorted(
            stats["tags_distribution"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]
        print(f"  热门标签: {dict(top_tags)}")

    print(f"\n{'=' * 60}")
    print(f"  总计: {total_seeds} 种子 + {total_variants} 变体 = {total_all} 测试用例")
    print(f"{'=' * 60}\n")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="AREval 数据集引导工具")
    parser.add_argument(
        "--domain",
        choices=list(SEED_FILES.keys()),
        help="指定生成的领域 (默认全部)",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=2,
        help="每个种子用例生成的变体数量 (默认 2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_project_root / ".areval" / "datasets"),
        help="输出目录 (默认 .areval/datasets/)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="仅列出种子数据集信息，不生成",
    )
    parser.add_argument(
        "--no-variants",
        action="store_true",
        help="只复制种子数据，不生成变体",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    print("AREval Dataset Bootstrap")
    print(f"Seed dir: {SEED_DIR}")
    print(f"Output:   {output_dir}")
    print()

    # 加载种子
    print("[1/4] Loading seed datasets...")
    seeds = load_seed(domain=args.domain)

    if not seeds:
        print("[ERROR] 没有找到任何种子数据集。请检查 datasets/seed/ 目录。")
        sys.exit(1)

    if args.list:
        print("\n[LIST] Seed dataset summary:")
        for domain, cases in seeds.items():
            tags = set()
            for tc in cases:
                tags.update(tc.tags)
            print(f"  {domain}: {len(cases)} cases, tags: {sorted(tags)}")
        sys.exit(0)

    # 生成变体
    print("\n[2/4] Generating variants (scale={})...".format(args.scale))
    results = {}
    for domain, seed_cases in seeds.items():
        if args.no_variants:
            variants = []
        else:
            variants = generate_variants(seed_cases, domain, scale=args.scale, seed=args.seed)
        print(f"  {domain}: {len(seed_cases)} seeds -> {len(variants)} variants")
        results[domain] = build_dataset(domain, seed_cases, variants)

    # 保存
    print(f"\n[3/4] Saving to {output_dir}...")
    for domain, data in results.items():
        filepath = save_dataset(data["cases"], domain, output_dir)
        print(f"  {domain}: {filepath} ({data['stats']['total_count']} cases)")

    # 报告
    print("\n[4/4] Generating report...")
    print_report(results)

    print("Done! 数据集已保存到 .areval/datasets/ 目录。")
    print("使用方式:")
    print(f'  from areval.datasets.manager import DatasetManager')
    print(f'  dm = DatasetManager()')
    for domain in results:
        filename = f"areval_{domain}_{datetime.now().strftime('%Y%m%d')}.jsonl"
        print(f'  ds = dm.create_from_file(".areval/datasets/{filename}", name="{domain}-eval")')


if __name__ == "__main__":
    main()
