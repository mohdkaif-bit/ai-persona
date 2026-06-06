"""
run_evals.py
Retrieval evaluation with correct metric definitions.

Metrics:
  Precision@K  = relevant chunks in top-K / K
  Recall@K     = relevant chunks in top-K / total relevant in corpus
  MRR          = mean(1 / rank_of_first_relevant_hit)
  Hit Rate@K   = fraction of questions with at least 1 relevant in top-K
"""

import json
from pathlib import Path
from backend.rag.retriever import retrieve

TOP_K = 5
golden_file = Path("evals/golden_set/questions.json")

with open(golden_file, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# ─────────────────────────────────────────
# Per-question accumulators
# ─────────────────────────────────────────
precision_at_1 = []
precision_at_3 = []
precision_at_5 = []

recall_at_1 = []
recall_at_3 = []
recall_at_5 = []

mrr_scores = []
hit_at_1 = []
hit_at_3 = []
hit_at_5 = []

results = []

for item in dataset:
    question = item["question"]
    expected = set(item["expected_sections"])

    retrieved = retrieve(query=question, top_k=TOP_K, use_reranker=True)
    retrieved_sections = [
        hit.get("metadata", {}).get("section", "unknown")
        for hit in retrieved
    ]

    # ── Precision@K = relevant in top-K / K ──
    def precision_at(k):
        top = retrieved_sections[:k]
        relevant = sum(1 for s in top if s in expected)
        return relevant / k if k > 0 else 0.0

    # ── Recall@K = relevant in top-K / total expected ──
    def recall_at(k):
        top = set(retrieved_sections[:k])
        relevant = len(top & expected) 
        return relevant / len(expected) if expected else 0.0

    # ── MRR = 1 / rank of first relevant hit ──
    mrr = 0.0
    for rank, sec in enumerate(retrieved_sections, start=1):
        if sec in expected:
            mrr = 1.0 / rank
            break

    # ── Hit Rate@K = at least 1 relevant in top-K ──
    def hit_at(k):
        return any(s in expected for s in retrieved_sections[:k])

    p1 = precision_at(1)
    p3 = precision_at(3)
    p5 = precision_at(5)

    r1 = recall_at(1)
    r3 = recall_at(3)
    r5 = recall_at(5)

    precision_at_1.append(p1)
    precision_at_3.append(p3)
    precision_at_5.append(p5)

    recall_at_1.append(r1)
    recall_at_3.append(r3)
    recall_at_5.append(r5)

    mrr_scores.append(mrr)

    h1 = hit_at(1)
    h3 = hit_at(3)
    h5 = hit_at(5)

    hit_at_1.append(h1)
    hit_at_3.append(h3)
    hit_at_5.append(h5)

    results.append({
        "question": question,
        "expected": list(expected),
        "retrieved": retrieved_sections,
        "precision@1": round(p1, 3),
        "precision@3": round(p3, 3),
        "precision@5": round(p5, 3),
        "recall@1": round(r1, 3),
        "recall@3": round(r3, 3),
        "recall@5": round(r5, 3),
        "mrr": round(mrr, 3),
        "hit@1": h1,
        "hit@3": h3,
        "hit@5": h5,
    })

# ─────────────────────────────────────────
# Aggregate
# ─────────────────────────────────────────
n = len(dataset)

def avg(lst): return sum(lst) / len(lst) if lst else 0.0

print("\n" + "=" * 50)
print("RETRIEVAL EVALUATION REPORT")
print("=" * 50)
print(f"Total Questions : {n}")
print()
print("── Precision@K (relevant retrieved / K) ──")
print(f"  Precision@1 : {avg(precision_at_1):.2%}")
print(f"  Precision@3 : {avg(precision_at_3):.2%}")
print(f"  Precision@5 : {avg(precision_at_5):.2%}")
print()
print("── Recall@K (relevant retrieved / total expected) ──")
print(f"  Recall@1    : {avg(recall_at_1):.2%}")
print(f"  Recall@3    : {avg(recall_at_3):.2%}")
print(f"  Recall@5    : {avg(recall_at_5):.2%}")
print()
print("── MRR (Mean Reciprocal Rank) ──")
print(f"  MRR         : {avg(mrr_scores):.4f}")
print()
print("── Hit Rate@K (at least 1 relevant in top-K) ──")
print(f"  Hit Rate@1  : {avg(hit_at_1):.2%}")
print(f"  Hit Rate@3  : {avg(hit_at_3):.2%}")
print(f"  Hit Rate@5  : {avg(hit_at_5):.2%}")

print()
print("=" * 50)
print("DETAILED RESULTS")
print("=" * 50)

for r in results:
    status = "PASS" if r["hit@1"] else "FAIL"
    print(f"\n{status} | {r['question']}")
    print(f"  Expected  : {r['expected']}")
    print(f"  Retrieved : {r['retrieved']}")
    print(f"  P@1={r['precision@1']}  P@3={r['precision@3']}  P@5={r['precision@5']}")
    print(f"  R@1={r['recall@1']}  R@3={r['recall@3']}  R@5={r['recall@5']}")
    print(f"  MRR={r['mrr']}")
    print("-" * 50)

# ─────────────────────────────────────────
# Save report
# ─────────────────────────────────────────
report = {
    "summary": {
        "total_questions": n,
        "precision@1": round(avg(precision_at_1), 4),
        "precision@3": round(avg(precision_at_3), 4),
        "precision@5": round(avg(precision_at_5), 4),
        "recall@1": round(avg(recall_at_1), 4),
        "recall@3": round(avg(recall_at_3), 4),
        "recall@5": round(avg(recall_at_5), 4),
        "mrr": round(avg(mrr_scores), 4),
        "hit_rate@1": round(avg(hit_at_1), 4),
        "hit_rate@3": round(avg(hit_at_3), 4),
        "hit_rate@5": round(avg(hit_at_5), 4),
    },
    "results": results
}

report_path = Path("evals/reports/retrieval_report.json")
report_path.parent.mkdir(parents=True, exist_ok=True)
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"\nReport saved to {report_path}")