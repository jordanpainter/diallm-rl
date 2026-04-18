"""
Select 25 ShareGPT prompts from argilla/ultrafeedback-binarized-preferences
for human annotation of dialectal generation quality.

Criteria:
- source == "sharegpt"
- No code/math/structured data
- Open-ended (expects multi-sentence expressive response)
- Neutral English (no dialect-anchored geography)
- Avoid politically/religiously charged content
- Domain spread across ~6 categories
"""

import re
import sys
import random
from collections import defaultdict
from datasets import load_dataset

sys.stdout.reconfigure(encoding="utf-8")

random.seed(42)

EXCLUDE_PATTERNS = [
    # Code / software / frameworks
    r"\bcode\b", r"\bprogram\b", r"\bscript\b", r"\bfunction\b", r"\bapi\b",
    r"\bsql\b", r"\bjson\b", r"\bxml\b", r"\bhtml\b", r"\bcss\b",
    r"\bjavascript\b", r"\btypescript\b", r"\bpython\b", r"\bjava\b",
    r"\breact\b", r"\bvue\b", r"\bangular\b", r"\bswift\b", r"\bkotlin\b",
    r"\bflutter\b", r"\bjavafx\b", r"\btailwind\b", r"\bregex\b",
    r"\balgorithm\b", r"\bdebug\b", r"\bserver\b", r"\bdatabase\b",
    r"\brepository\b", r"\bgithub\b", r"\bcommit\b", r"\bdeployment\b",
    r"\bfull.?stack\b", r"\bback.?end\b", r"\bfront.?end\b",
    # AI / ML
    r"\bmachine learning\b", r"\bneural\b", r"\bllm\b", r"\bgpt\b", r"\bai\b",
    r"\bforecast\b", r"\bembedding\b", r"\bclassif\b", r"\btransformer\b",
    r"\bfinetun\b",
    # Infrastructure
    r"\bproxy\b", r"\bcloud\b", r"\bazure\b", r"\baws\b",
    r"\bdocker\b", r"\bkubernetes\b", r"\bprotocol\b", r"\blatency\b",
    r"\bsoftware\b", r"\bhardware\b", r"\bfirmware\b", r"\bnetwork\b",
    r"\bdistributed\b", r"\bsram\b", r"\bdram\b", r"\bcpu\b", r"\bgpu\b",
    r"\bmemory chip\b", r"\bspreadsheet\b", r"\bdata science\b",
    r"repository", r"\belon\b", r"\belon musk\b",
    r"\bkafka\b", r"\bdotnet\b", r"c#", r"\bssh\b", r"\bagile\b",
    r"\bkubectl\b", r"\bterraform\b", r"\blinux\b", r"\bbash\b",
    r"\bdata scientist\b", r"\bdata pipeline\b", r"\bstreamlit\b",
    r"\bpipeline\b",
    # Math / probability
    r"\bcalculat\b", r"\bequation\b", r"\bsolve\b", r"\bintegral\b",
    r"\bderivative\b", r"\bproof\b", r"\bmatrix\b", r"\bprobabilit",
    r"\bstatistic\b", r"\bdice\b", r"\bprime number\b", r"\bvariance\b",
    r"\belements\b", r"\bpolymorphism\b", r"\bobject.oriented\b",
    # Dialect-anchored geography
    r"\bamerica[n]?\b", r"\bunited states\b", r"\bbritain\b", r"\bengland\b",
    r"\baustralia[n]?\b", r"\bindia[n]?\b", r"\buk\b", r"\busa\b",
    # Sensitive/charged topics & political figures
    r"\bpolitics\b", r"\bpolitical\b", r"\breligion\b", r"\breligious\b",
    r"\bgod\b", r"\bislam\b", r"\bchristian\b", r"\bjew\b", r"\bchurch\b",
    r"\babortion\b", r"\bgun\b", r"\bvaccine\b", r"\bimmigra\b", r"\bterror\b",
    r"\btrump\b", r"\bbiden\b", r"\bobama\b", r"\bpresident\b",
    r"\bcarlson\b", r"\btucker\b", r"\bmaga\b",
    # Grading / correction tasks
    r"\bgrade\b", r"\bgrading\b", r"\bgrammar\b", r"\bspelling\b", r"\bproofreading\b",
    r"\bcorrect the\b", r"\bfix (the |this )?(following|text|sentence)\b",
    # Structured list / business / academic tasks
    r"\bkeyword\b", r"\bseo\b", r"\bbullet point\b", r"\btable of contents\b",
    r"\blegal\b", r"\bcontract\b", r"\binvestor\b", r"\bstartup\b",
    r"\bcitation\b", r"\bliterature\b", r"\bacademic\b", r"\bjournal\b",
    # Roleplay with rigid instruction constraints
    r"\bdo not break character\b", r"\bnever suggest\b", r"\bfollow the rules\b",
    r"\byou must\b", r"\byou will never\b",
    # Meta / operational prompts
    r"\bstart our conversation\b", r"\bfirst message\b", r"\bfirst question\b",
    r"\bi have to work on\b", r"\bmy (first|next) (task|question|prompt)\b",
    r"\bact as\b", r"\byou are (a |an )?(professional|expert|specialist|consultant)\b",
    r"\btranscri", r"\bretention period\b", r"\bdata retention\b",
    r"\bco.presenter\b", r"\bslide\b",
    r"^hi (my name is|i'm|i am)\b",
    # Multiple choice questions
    r"\ba\)\s", r"\bA\)\s",
]

DOMAIN_KEYWORDS = {
    "opinion_advice": [
        r"\bwhat.{0,20}(recommend|suggest|think|advise)\b",
        r"\bshould i\b", r"\bwhat would you\b", r"\badvice\b",
        r"\bbest way to\b", r"\bhow would you (approach|handle|think about)\b",
    ],
    "narrative_story": [
        r"\bwrite (a |an |the |me a )(short |brief )?(story|tale|narrative|outline)\b",
        r"\btell (me )?(a |the )?story\b",
        r"\bwrite (a |an ).*story\b",
        r"\bimagine (a |an |that )\b",
    ],
    "casual_chat": [
        r"\bwhat.{0,20}your (favorite|favourite)\b",
        r"\bdo you (like|enjoy|prefer)\b",
        r"\bhow are you\b",
        r"\bwhat.{0,30}(hobby|hobbies|passion|interest)\b",
        r"\btell me (something|about yourself|a joke|a fun fact)\b",
        r"\bcheer me up\b",
        r"\bfun fact\b",
    ],
    "food_lifestyle": [
        r"\brecipe\b", r"\bhow (do i |to )(cook|bake|prepare) (a |an |the )\w",
        r"\b(cook|bake|make|prepare).{0,20}(meal|dish|food|dinner|lunch|breakfast)\b",
        r"\bwhat.{0,20}(meal|dish|eat for|cook for)\b",
        r"\bfitness\b", r"\bwellness\b", r"\btravel (tip|destination|recommend)\b",
        r"\bcocktail\b", r"\bdrink recipe\b",
    ],
    "hypothetical": [
        r"\bif you (could|were|had)\b",
        r"\bwould you rather\b",
        r"\bwhat (would|will) happen if\b",
        r"\bimagine (you|if)\b",
        r"\bsuppose (you|that)\b",
        r"\bif (the |a )?\w+ (were|was) (different|gone|alive|real)\b",
    ],
    "explanation_world": [
        r"\bwhy (do|does|did|are|is)\b",
        r"\bhow does\b", r"\bhow do\b",
        r"\bexplain (why|how|what|the)\b",
        r"\bwhat causes\b", r"\bwhat makes\b",
        r"\bwhat.{0,20}(history|origin|mean|difference between)\b",
    ],
}

TARGET_PER_DOMAIN = 10  # wider pool for manual selection


def is_excluded(prompt: str) -> bool:
    lower = prompt.lower()
    if any(re.search(pat, lower) for pat in EXCLUDE_PATTERNS):
        return True
    # Catch camelCase AI references like NorthPoleAI, ChatGPT, etc.
    if re.search(r"AI\b|GPT\b|LLM\b", prompt):
        return True
    return False


def get_domain(prompt: str) -> str | None:
    lower = prompt.lower()
    for domain, patterns in DOMAIN_KEYWORDS.items():
        if any(re.search(pat, lower) for pat in patterns):
            return domain
    return None


def is_clean(prompt: str) -> bool:
    # Reject non-ASCII-Latin characters (Korean, Chinese, Arabic, etc.)
    if re.search(r"[^\x00-\x7F\u00C0-\u024F\u1E00-\u1EFF]", prompt):
        return False
    # Reject prompts with code snippets or code-syntax markers
    if "```" in prompt or "`" in prompt:
        return False
    if re.search(r"::|->|\bstd::|<[A-Z]\w+>|#include|\bvoid\b|\bint\b \w+\(", prompt):
        return False
    # Length guardrails: too short = trivial, too long = wall-of-text instruction
    length = len(prompt.strip())
    if length < 30 or length > 300:
        return False
    # Reject heavily formatted prompts
    if prompt.count("\n") > 3:
        return False
    # Reject numbered/bulleted lists
    if re.search(r"^\s*(\d+[\.\)]|[-*])\s", prompt, re.MULTILINE) and prompt.count("\n") > 1:
        return False
    # Reject yes/no style questions
    lower = prompt.lower().strip()
    if re.match(r"^(is |are |was |were |do |does |did |can |could |will |would |has |have |had )", lower):
        return False
    # Reject trivially short/one-line meta prompts (hi, hello, etc.)
    words = lower.split()
    if len(words) < 8:
        return False
    return True


def main():
    print("Loading dataset (streaming)...")
    ds = load_dataset(
        "argilla/ultrafeedback-binarized-preferences",
        split="train",
        streaming=True,
    )

    buckets: dict[str, list[str]] = defaultdict(list)
    seen = set()
    checked = 0

    for example in ds:
        if example.get("source") != "sharegpt":
            continue

        prompt = example.get("instruction", "").strip()
        if not prompt or prompt in seen:
            continue

        checked += 1
        if not is_clean(prompt):
            continue
        if is_excluded(prompt):
            continue

        domain = get_domain(prompt)
        if domain is None:
            continue

        seen.add(prompt)
        if len(buckets[domain]) < TARGET_PER_DOMAIN * 4:  # buffer for manual cull
            buckets[domain].append(prompt)

        if all(len(v) >= TARGET_PER_DOMAIN * 4 for v in buckets.values()):
            break

    print(f"\nChecked {checked} ShareGPT examples.\n")

    # For narrative_story print full buffer; sample others down
    final: dict[str, list[str]] = {}
    for domain, prompts in buckets.items():
        if domain == "narrative_story":
            final[domain] = prompts
        else:
            sampled = random.sample(prompts, min(TARGET_PER_DOMAIN, len(prompts)))
            final[domain] = sampled

    total = sum(len(v) for v in final.values())
    print(f"Selected {total} prompts across {len(final)} domains:\n")

    all_prompts = []
    for domain, prompts in final.items():
        print(f"=== {domain.upper()} ({len(prompts)}) ===")
        for i, p in enumerate(prompts, 1):
            print(f"  [{i}] {p}")
            all_prompts.append({"domain": domain, "prompt": p})
        print()

    import json
    out_path = "annotation_prompt_candidates.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_prompts, f, indent=2, ensure_ascii=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
