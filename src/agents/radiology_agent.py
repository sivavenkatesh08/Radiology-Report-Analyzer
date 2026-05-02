"""
Radiology Agent Module
-----------------------
Agent-based reasoning system that combines:
  1. Full text analysis (rule-based interpretation)
  2. RAG retrieval (semantic knowledge lookup)
  3. Decision fusion (merges both sources into final output)

The agent processes the FULL report text (not summary) and produces:
  - Explanation (clinical interpretation)
  - Severity (Low / Medium / High)
  - Recommendation (actionable next steps)
  - Key Findings (structured list)
  - RAG Context (retrieved medical knowledge)
"""

import logging

logger = logging.getLogger(__name__)


def _classify_severity_from_score(score):
    """Convert numeric severity score to categorical label."""
    if score >= 2.5:
        return "🔴 High"
    elif score >= 1.5:
        return "🟡 Medium"
    else:
        return "🟢 Low"


def _severity_to_score(severity_text):
    """Convert severity text to numeric score for comparison."""
    text = severity_text.lower().strip()
    if "high" in text:
        return 3.0
    elif "medium to high" in text:
        return 2.5
    elif "medium" in text:
        return 2.0
    elif "low to medium" in text:
        return 1.5
    elif "low" in text:
        return 0.5
    return 1.0


def analyze(full_text):
    """
    Run the full agent pipeline on a radiology report.

    Steps:
        1. Rule-based interpretation (interpreter module)
        2. RAG retrieval from FAISS knowledge base
        3. Decision fusion — combine both into final output

    Args:
        full_text: The FULL report text (not summary).

    Returns:
        dict with keys:
            explanation, severity, recommendation, key_findings,
            rag_context, rag_available
    """
    if not full_text or not full_text.strip():
        return {
            "explanation": "No report text provided for analysis.",
            "severity": "⚪ Unknown",
            "recommendation": "Please provide a valid radiology report.",
            "key_findings": [],
            "rag_context": [],
            "rag_available": False,
        }

    # ── Step 1: Rule-based interpretation ──
    from src.nlp.interpreter import interpret_report
    rule_result = interpret_report(full_text)

    # ── Step 2: RAG retrieval ──
    rag_results = []
    rag_available = False
    try:
        from src.rag.vector_store import retrieve, is_available
        rag_available = is_available()
        if rag_available:
            rag_results = retrieve(full_text, top_k=3)
            logger.info(f"RAG retrieved {len(rag_results)} documents.")
    except Exception as e:
        logger.warning(f"RAG retrieval failed: {e}")

    # ── Step 3: Decision fusion ──
    return _fuse_decisions(rule_result, rag_results, rag_available)


def _fuse_decisions(rule_result, rag_results, rag_available):
    """
    Fuse rule-based and RAG-based outputs into a unified clinical decision.

    Strategy:
        - Explanation: Rule-based findings + RAG medical context
        - Severity: Highest severity from either source
        - Recommendation: Rule-based + RAG-augmented guidance
        - Key Findings: From rule-based analysis
        - RAG Context: Retrieved topics for transparency
    """

    # ── Explanation ──
    explanation_parts = [rule_result["explanation"]]

    if rag_results:
        # Add RAG-sourced context as supplementary medical knowledge
        rag_context_text = "\n\n📚 Medical Knowledge Context (RAG):"
        for i, doc in enumerate(rag_results[:3], 1):
            rag_context_text += f"\n• [{doc['topic']}] — Severity guidance: {doc['severity']}"
        explanation_parts.append(rag_context_text)

    combined_explanation = "\n".join(explanation_parts)

    # ── Severity ── (take the highest from both sources)
    rule_severity = rule_result["severity"]
    max_severity_score = 0

    # Score from rule-based
    if "High" in rule_severity:
        max_severity_score = 3.0
    elif "Medium" in rule_severity:
        max_severity_score = 2.0
    elif "Low" in rule_severity:
        max_severity_score = 0.5

    # Check RAG severity
    if rag_results:
        for doc in rag_results[:2]:  # top 2 most relevant
            rag_score = _severity_to_score(doc.get("severity", ""))
            # Only escalate if RAG is highly relevant (score > 0.5)
            if doc.get("score", 0) > 0.5 and rag_score > max_severity_score:
                max_severity_score = rag_score

    final_severity = _classify_severity_from_score(max_severity_score)

    # ── Recommendation ── (combine rule + RAG)
    recommendation_parts = [rule_result["recommendation"]]

    if rag_results and rag_results[0].get("score", 0) > 0.4:
        top_rag = rag_results[0]
        # Only add RAG recommendation if it provides additional value
        rag_rec = top_rag.get("recommendation", "")
        if rag_rec and rag_rec not in rule_result["recommendation"]:
            recommendation_parts.append(
                f"[RAG-augmented] {rag_rec}"
            )

    combined_recommendation = " ".join(recommendation_parts)

    # ── RAG Context metadata ──
    rag_context = []
    for doc in rag_results:
        rag_context.append({
            "topic": doc["topic"],
            "relevance": f"{doc['score']:.2f}",
            "severity": doc["severity"],
        })

    return {
        "explanation": combined_explanation,
        "severity": final_severity,
        "recommendation": combined_recommendation,
        "key_findings": rule_result.get("key_findings", []),
        "rag_context": rag_context,
        "rag_available": rag_available,
    }
