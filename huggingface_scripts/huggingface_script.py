import os
import json
import argparse
from datetime import datetime
import csv
csv.field_size_limit(2_000_000_000)
from collections import defaultdict

parser = argparse.ArgumentParser(description="Process leaderboard results for HuggingFace")
parser.add_argument("--verbose", "-v", action="store_true", help="Print progress messages")
args = parser.parse_args()

# ===========================
# Your mappings
# ===========================
# <32B is considered small
model_size_map = {
    "ai21labs/jamba-large-1.7": "large",
    "ai21labs/jamba-mini-1.7": "large",
    "ai21labs/jamba-mini-2": "large",
    "amazon/nova-lite-v1:0": "unknown",
    "amazon/nova-micro-v1:0": "unknown",
    "amazon/nova-pro-v1:0": "unknown",
    "amazon/nova-2-lite-v1:0": "unknown",
    "antgroup/finix_s1_32b": "small",
    "antgroup/antfinix-a1": "large",
    "anthropic/claude-haiku-4-5-20251001": "large",
    "anthropic/claude-opus-4-1-20250805": "large",
    "anthropic/claude-opus-4-20250514": "large",
    "anthropic/claude-sonnet-4-20250514": "large",
    "anthropic/claude-sonnet-4-5-20250929": "large",
    "anthropic/claude-opus-4-6": "large",
    "anthropic/claude-opus-4-5": "large",
    "anthropic/claude-opus-4": "large",
    "anthropic/claude-sonnet-4": "large",
    "anthropic/claude-sonnet-4-5": "large",
    "anthropic/claude-haiku-4-5": "small",
    "anthropic/claude-opus-4-1": "large",
    "arcee-ai/trinity-large-preview": "large",
    "CohereLabs/c4ai-aya-expanse-32b": "small",
    "CohereLabs/c4ai-aya-expanse-8b": "small",
    "CohereLabs/command-a-03-2025": "large",
    "CohereLabs/command-r-plus-08-2024": "large",
    "CohereLabs/command-a": "large",
    "CohereLabs/command-r-plus": "large",
    "deepseek-ai/DeepSeek-V3": "large",
    "deepseek-ai/DeepSeek-V3.1": "large",
    "deepseek-ai/DeepSeek-V3.2-Exp": "large",
    "deepseek-ai/DeepSeek-R1": "large",
    "deepseek-ai/DeepSeek-V3.2": "large",
    "google/gemini-2.5-flash": "large", #?
    "google/gemini-2.5-flash-lite": "large", #?
    "google/gemini-2.5-pro": "large",
    "google/gemini-3-pro-preview": "large",
    "google/gemma-3-4b-it": "small",
    "google/gemma-3-12b-it": "small",
    "google/gemma-3-27b-it": "small",
    "google/gemini-3-flash-preview": "large",
    "ibm-granite/granite-4.0-h-small": "small",
    "ibm-granite/granite-3.3-8b-instruct": "small",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": "large",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": "large",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": "large",
    "microsoft/Phi-4": "small",
    "microsoft/Phi-4-mini-instruct": "small",
    "MiniMaxAI/minimax-m2p1": "large",
    "mistralai/ministral-3b-2410": "small",
    "mistralai/ministral-8b-2410": "small",
    "mistralai/mistral-large-2411": "large",
    "mistralai/mistral-medium-2508": "large",
    "mistralai/mistral-small-2501": "small",
    "mistralai/ministral-8b": "small",
    "mistralai/ministral-14b": "small",
    "mistralai/ministral-3b": "small",
    "mistralai/mistral-medium": "large",
    "mistralai/mistral-large": "large",
    "mistralai/mistral-small": "small",
    "moonshotai/Kimi-K2-Instruct-0905": "large",
    "moonshotai/Kimi-K2-Instruct": "large",
    "moonshotai/Kimi-K2.5": "large",
    "nvidia/Nemotron-3-Nano-30B-A3B": "small",
    "openai/gpt-5-high-2025-08-07": "large",
    "openai/gpt-4.1-2025-04-14": "large",
    "openai/gpt-4o-2024-08-06": "large",
    "openai/gpt-5-mini-2025-08-07": "large",
    "openai/gpt-5-minimal-2025-08-07": "large",
    "openai/gpt-5-nano-2025-08-07": "large", # ?
    "openai/gpt-5.2-high": "large",
    "openai/gpt-5.2-low": "large",
    "openai/gpt-oss-120b": "large",
    "openai/o3-pro": "large",
    "openai/o4-mini-high-2025-04-16": "large", # ?
    "openai/o4-mini-low-2025-04-16": "large", # ?
    "openai/gpt-5-mini": "unknown",
    "openai/gpt-4.1": "large",
    "openai/gpt-5-minimal": "large",
    "openai/o4-mini-low": "unknown",
    "openai/gpt-5.1-high": "large",
    "openai/gpt-5-nano": "unknown",
    "openai/o4-mini-high": "unknown",
    "openai/gpt-4o": "large",
    "openai/gpt-5-high": "large",
    "openai/gpt-5.1-low": "large",
    "qwen/qwen3-4b": "small",
    "qwen/qwen3-8b": "small",
    "qwen/qwen3-14b": "small",
    "qwen/qwen3-32b": "small",
    "qwen/qwen3-235b-a22b": "large",
    "qwen/qwen3-next-80b-a3b-thinking": "large",
    "snowflake/snowflake-arctic-instruct": "large",
    "vectara/mockingbird-2.0": "small",
    "xai-org/grok-3": "large",
    "xai-org/grok-4-fast-non-reasoning": "large",
    "xai-org/grok-4-fast-reasoning": "large",
    "xai-org/grok-4-1-fast-non-reasoning": "large",
    "xai-org/grok-4-1-fast-reasoning": "large",
    "zai-org/glm-4p7-flash": "small",
    "zai-org/glm-4p7": "large",
    "zai-org/GLM-4.5-AIR-FP8": "large",
    "zai-org/GLM-4.6": "large",

}

accessibility_map = {
    "ai21labs/jamba-large-1.7": "open",
    "ai21labs/jamba-mini-1.7": "open",
    "ai21labs/jamba-mini-2": "open",
    "amazon/nova-lite-v1:0": "commercial",
    "amazon/nova-micro-v1:0": "commercial",
    "amazon/nova-pro-v1:0": "commercial",
    "amazon/nova-2-lite-v1:0": "commercial",
    "antgroup/finix_s1_32b": "commercial",
    "antgroup/antfinix-a1": "commercial",
    "anthropic/claude-haiku-4-5-20251001": "commercial",
    "anthropic/claude-opus-4-1-20250805": "commercial",
    "anthropic/claude-opus-4-20250514": "commercial",
    "anthropic/claude-sonnet-4-20250514": "commercial",
    "anthropic/claude-sonnet-4-5-20250929": "commercial",
    "anthropic/claude-opus-4-6": "commercial",
    "anthropic/claude-opus-4-5": "commercial",
    "anthropic/claude-opus-4": "commercial",
    "anthropic/claude-sonnet-4": "commercial",
    "anthropic/claude-sonnet-4-5": "commercial",
    "anthropic/claude-haiku-4-5": "commercial",
    "anthropic/claude-opus-4-1": "commercial",
    "arcee-ai/trinity-large-preview": "open",
    "CohereLabs/c4ai-aya-expanse-32b": "open",
    "CohereLabs/c4ai-aya-expanse-8b": "open",
    "CohereLabs/command-a-03-2025": "open",
    "CohereLabs/command-r-plus-08-2024": "open",
    "CohereLabs/command-a": "open",
    "CohereLabs/command-r-plus": "open",
    "deepseek-ai/DeepSeek-V3": "open",
    "deepseek-ai/DeepSeek-V3.1": "open",
    "deepseek-ai/DeepSeek-V3.2-Exp": "open",
    "deepseek-ai/DeepSeek-R1": "open",
    "deepseek-ai/DeepSeek-V3.2": "open",
    "google/gemini-2.5-flash": "commercial",
    "google/gemini-3-flash-preview": "commercial",
    "google/gemini-2.5-flash-lite": "commercial",
    "google/gemini-2.5-pro": "commercial",
    "google/gemini-3-pro-preview": "commercial",
    "google/gemma-3-4b-it": "open",
    "google/gemma-3-12b-it": "open",
    "google/gemma-3-27b-it": "open",
    "ibm-granite/granite-4.0-h-small": "open",
    "ibm-granite/granite-3.3-8b-instruct": "open",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": "open",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": "open",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": "open",
    "microsoft/Phi-4": "open",
    "microsoft/Phi-4-mini-instruct": "open",
    "MiniMaxAI/minimax-m2p1": "open",
    "mistralai/ministral-3b-2410": "open", # ?
    "mistralai/ministral-8b-2410": "open",
    "mistralai/mistral-large-2411": "open",
    "mistralai/mistral-medium-2508": "commercial",
    "mistralai/mistral-small-2501": "open",
    "mistralai/ministral-8b": "open",
    "mistralai/ministral-14b": "open",
    "mistralai/ministral-3b": "open",
    "mistralai/mistral-medium": "commercial",
    "mistralai/mistral-large": "open",
    "mistralai/mistral-small": "open",
    "moonshotai/Kimi-K2-Instruct-0905": "open",
    "moonshotai/Kimi-K2-Instruct": "open",
    "moonshotai/Kimi-K2.5": "open",
    "nvidia/Nemotron-3-Nano-30B-A3B": "open",
    "openai/gpt-5-high-2025-08-07": "commercial",
    "openai/gpt-4.1-2025-04-14": "commercial",
    "openai/gpt-4o-2024-08-06": "commercial",
    "openai/gpt-5-mini-2025-08-07": "commercial",
    "openai/gpt-5-minimal-2025-08-07": "commercial",
    "openai/gpt-5-nano-2025-08-07": "commercial",
    "openai/gpt-oss-120b": "open",
    "openai/gpt-5.2-high": "commercial",
    "openai/gpt-5.2-low": "commercial",
    "openai/o3-pro": "commercial",
    "openai/o4-mini-high-2025-04-16": "commercial",
    "openai/o4-mini-low-2025-04-16": "commercial",
    "openai/gpt-5-mini": "commercial",
    "openai/gpt-4.1": "commercial",
    "openai/gpt-5-minimal": "commercial",
    "openai/o4-mini-low": "commercial",
    "openai/gpt-5.1-high": "commercial",
    "openai/gpt-5-nano": "commercial",
    "openai/o4-mini-high": "commercial",
    "openai/gpt-4o": "commercial",
    "openai/gpt-5-high": "commercial",
    "openai/gpt-5.1-low": "commercial",
    "qwen/qwen3-32b": "open",
    "qwen/qwen3-next-80b-a3b-thinking": "open",
    "qwen/qwen3-4b": "open",
    "qwen/qwen3-8b": "open",
    "qwen/qwen3-14b": "open",
    "qwen/qwen3-235b-a22b": "open",
    "snowflake/snowflake-arctic-instruct": "open",
    "vectara/mockingbird-2.0": "commercial",
    "xai-org/grok-3": "commercial",
    "xai-org/grok-4-fast-non-reasoning": "commercial",
    "xai-org/grok-4-fast-reasoning": "commercial",
    "xai-org/grok-4-1-fast-non-reasoning": "commercial",
    "xai-org/grok-4-1-fast-reasoning": "commercial",
    "zai-org/glm-4p7": "open",
    "zai-org/glm-4p7-flash": "open",
    "zai-org/GLM-4.5-AIR-FP8": "open",
    "zai-org/GLM-4.6": "open",

}

VALID_SIZES = {"small", "large", "unknown"}
VALID_ACCESS = {"open", "commercial", "unknown"}

# Input and output base directories
input_base = "../output"
output_base = "hf_structured_output"

# Ensure output directory exists
os.makedirs(output_base, exist_ok=True)

# Get set of already processed models (company/model folders that exist in output)
existing_models = set()
if os.path.exists(output_base):
    for company_dir in os.listdir(output_base):
        company_path = os.path.join(output_base, company_dir)
        if os.path.isdir(company_path):
            for model_dir in os.listdir(company_path):
                model_path = os.path.join(company_path, model_dir)
                if os.path.isdir(model_path):
                    existing_models.add(f"{company_dir}/{model_dir}")

# Load leaderboard CSV (article_id → metadata)
lb_path = "../datasets/leaderboard_dataset_v2.csv"
lb_data = {}
categories_set = set()
text_type_map = {
    "standard_text": "low_complexity_text",
    "intensive_text": "high_complexity_text"
}

def round_floats(obj, ndigits=3):
    if isinstance(obj, float):
        return round(obj, ndigits)
    elif isinstance(obj, dict):
        return {k: round_floats(v, ndigits) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(v, ndigits) for v in obj]
    return obj

with open(lb_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        aid = int(row["article_id"])
        category = row["category"]
        text_type = row["text_type"]

        # Build mapping
        lb_data[aid] = {
            "category": category,
            "text_complexity": text_type_map.get(text_type, "unknown")
        }

        categories_set.add(category)

# Traverse directories for stats.jsonl
for root, dirs, files in os.walk(input_base):
    if "stats.jsonl" not in files:
        continue

    stats_path = os.path.join(root, "stats.jsonl")
    summaries_path = os.path.join(root, "summaries.jsonl")
    judgments_path = os.path.join(root, "judgments.jsonl")

    # Read stats.jsonl
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.loads(f.readline().strip())

    company = stats.get("company")
    model_name = stats.get("model_name")
    date_code = stats.get("date_code")

    # Skip if this model already has results in output
    model_key = f"{company}/{model_name}-{date_code}"
    if model_key in existing_models:
        if args.verbose:
            print(f"Skipping {model_key} (already processed)")
        continue

    if args.verbose:
        print(f"Processing {model_key}...")

    hallucination_rate_total = stats.get("hallucination_rate", 0.0)
    answer_rate_total = stats.get("answer_rate", 0.0)
    avg_word_count_total = stats.get("avg_word_count", 0.0)

    combined_name = f"{company}/{model_name}"

    # Lookup size
    if combined_name in model_size_map:
        model_size = model_size_map[combined_name]
    else:
        print(f"WARNING: model_size not found for '{combined_name}'. Using 'unknown'.")
        model_size = "unknown"

    if model_size not in VALID_SIZES:
        print(f"WARNING: Invalid model_size '{model_size}' for {combined_name}. Forcing 'unknown'.")
        model_size = "unknown"

    # Lookup accessibility
    if combined_name in accessibility_map:
        accessibility = accessibility_map[combined_name]
    else:
        print(f"WARNING: accessibility not found for '{combined_name}'. Using 'unknown'.")
        accessibility = "unknown"

    if accessibility not in VALID_ACCESS:
        print(f"WARNING: Invalid accessibility '{accessibility}' for {combined_name}. Forcing 'unknown'.")
        accessibility = "unknown"

    # -------------------------
    # Load summaries AND judgments
    # -------------------------
    summaries = {}
    if os.path.exists(summaries_path):
        with open(summaries_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                summaries[item["summary_uid"]] = item

    judgments = []
    if os.path.exists(judgments_path):
        with open(judgments_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                judgments.append(json.loads(line))

    # -------------------------
    # Grouping containers
    # -------------------------
    cat_stats = defaultdict(lambda: {"valid": 0, "consistent": 0, "word_total": 0})
    complexity_stats = defaultdict(lambda: {"valid": 0, "consistent": 0, "word_total": 0})

    # -------------------------
    # Process judgments (only valid ones)
    # -------------------------
    for j in judgments:
        if not j.get("is_valid", False):
            continue

        uid = j["summary_uid"]
        if uid not in summaries:
            continue

        summ = summaries[uid]
        aid = summ["article_id"]

        if aid not in lb_data:
            continue

        category = lb_data[aid]["category"]
        complexity_label = lb_data[aid]["text_complexity"]

        # Word count
        summary_text = summ.get("summary", "")
        wc = len(summary_text.split())

        # Valid judgment
        cat_stats[category]["valid"] += 1
        cat_stats[category]["word_total"] += wc

        complexity_stats[complexity_label]["valid"] += 1
        complexity_stats[complexity_label]["word_total"] += wc

        # Factual consistency → hallucination logic
        score = j.get("hhem_score", 0)
        if score >= 0.5:
            cat_stats[category]["consistent"] += 1
            complexity_stats[complexity_label]["consistent"] += 1

    # -------------------------
    # Compute final grouped metrics
    # -------------------------
    category_results = {}
    for cat in sorted(categories_set):
        data = cat_stats[cat]
        valid = data["valid"]

        if valid == 0:
            category_results[cat] = {
                "hallucination_rate": 0.0,
                "answer_rate": 0.0,
                "average_summary_length": 0.0
            }
            continue

        factual_rate = (data["consistent"] / valid) * 100
        halluc_rate = 100 - factual_rate
        answer_rate = (valid / valid) * 100  # valid summaries / valid summaries → always 100%?

        # Clarification: answer rate should be valid_judgments / total_summary_count_in_category.
        # But you said: "Supply the answer rate using is_valid". So we interpret as % of summaries with valid judgments.
        # Need denominator = count of summaries in this category.
        # Let's compute:

        # Count total summaries in this category:
        total_summaries_cat = sum(1 for s in summaries.values()
                                  if s["article_id"] in lb_data
                                  and lb_data[s["article_id"]]["category"] == cat)

        if total_summaries_cat > 0:
            answer_rate = (valid / total_summaries_cat) * 100
        else:
            answer_rate = 0.0

        avg_len = data["word_total"] / valid if valid > 0 else 0.0

        category_results[cat] = {
            "hallucination_rate": halluc_rate,
            "answer_rate": answer_rate,
            "average_summary_length": avg_len
        }

    # Complexity results
    text_complexity_results = {}
    for comp_label, data in complexity_stats.items():
        valid = data["valid"]

        # Count total summaries for denominator
        total_summaries_comp = sum(
            1 for s in summaries.values()
            if s["article_id"] in lb_data
            and lb_data[s["article_id"]]["text_complexity"] == comp_label
        )

        if valid == 0 or total_summaries_comp == 0:
            text_complexity_results[comp_label] = {
                "hallucination_rate": 0.0,
                "answer_rate": 0.0,
                "average_summary_length": 0.0
            }
            continue

        factual_rate = (data["consistent"] / valid) * 100
        halluc_rate = 100 - factual_rate
        answer_rate = (valid / total_summaries_comp) * 100
        avg_len = data["word_total"] / valid

        text_complexity_results[comp_label] = {
            "hallucination_rate": halluc_rate,
            "answer_rate": answer_rate,
            "average_summary_length": avg_len
        }

    # -------------------------
    # Build final JSON output
    # -------------------------
    now = datetime.now()
    timestamp = now.strftime("results_%Y-%m-%d %H:%M:%S.%f.json")

    factual_consistency_rate_total = 100.0 - hallucination_rate_total

    new_data = {
        "config": {
            "model_dtype": "float16",
            "model_name": f"{company}/{model_name}-{date_code}",
            "model_sha": "main",
        },
        "results": {
            "hallucination_rate": {
                "hallucination_rate": hallucination_rate_total
            },
            "factual_consistency_rate": {
                "factual_consistency_rate": factual_consistency_rate_total
            },
            "answer_rate": {
                "answer_rate": answer_rate_total
            },
            "average_summary_length": {
                "average_summary_length": avg_word_count_total
            }
        },
        "model_annotations": {
            "model_size": model_size,
            "accessibility": accessibility
        },
        # NEW SECTIONS:
        "category_results": category_results,
        "text_complexity_results": text_complexity_results,
    }

    # Create output folder
    relative_path = os.path.relpath(root, input_base)
    output_dir = os.path.join(output_base, relative_path)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, timestamp)

    new_data = round_floats(new_data, ndigits=3)

    # Save file
    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(new_data, out_f, indent=2)