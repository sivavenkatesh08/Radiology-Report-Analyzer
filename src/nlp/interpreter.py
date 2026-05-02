"""
Medical Report Interpreter Module
---------------------------------
Performs rule-based clinical interpretation on radiology report text.
Uses FULL report text (not summary) for comprehensive analysis.
Returns structured output: explanation, severity, recommendation, key_findings.
"""

import re


# ────────────────────────────────────────────────────────────────
# Medical keyword database with severity weights and explanations
# ────────────────────────────────────────────────────────────────

MEDICAL_RULES = [
    # ── HIGH SEVERITY ──
    {
        "keywords": ["mass", "tumor", "neoplasm", "malignant", "metastasis", "metastatic"],
        "explanation": "Suspicious mass or tumor identified. Potential malignancy detected.",
        "severity": 3,
        "recommendation": "Urgent oncology referral recommended. Consider biopsy and advanced imaging (PET/CT).",
        "finding": "Mass / Tumor",
    },
    {
        "keywords": ["pneumothorax", "collapsed lung"],
        "explanation": "Pneumothorax (collapsed lung) detected.",
        "severity": 3,
        "recommendation": "Immediate medical intervention required. Chest tube placement may be necessary.",
        "finding": "Pneumothorax",
    },
    {
        "keywords": ["pulmonary embolism", "embolism", "pe "],
        "explanation": "Pulmonary embolism suspected.",
        "severity": 3,
        "recommendation": "Emergency treatment required. CT pulmonary angiography and anticoagulation therapy recommended.",
        "finding": "Pulmonary Embolism",
    },
    {
        "keywords": ["hemorrhage", "bleeding", "haemorrhage"],
        "explanation": "Active hemorrhage or bleeding identified.",
        "severity": 3,
        "recommendation": "Urgent intervention required. Consult interventional radiology or surgery.",
        "finding": "Hemorrhage",
    },
    {
        "keywords": ["aortic dissection", "dissection"],
        "explanation": "Aortic dissection suspected — life-threatening condition.",
        "severity": 3,
        "recommendation": "Emergency surgical consultation required immediately.",
        "finding": "Aortic Dissection",
    },
    {
        "keywords": ["large effusion", "massive effusion", "significant effusion"],
        "explanation": "Large pleural effusion detected causing potential respiratory compromise.",
        "severity": 3,
        "recommendation": "Thoracentesis may be required. Urgent pulmonology consultation recommended.",
        "finding": "Large Pleural Effusion",
    },

    # ── MEDIUM-HIGH SEVERITY ──
    {
        "keywords": ["consolidation", "airspace disease", "lobar pneumonia"],
        "explanation": "Lung consolidation detected, consistent with pneumonia or airspace disease.",
        "severity": 2.5,
        "recommendation": "Start empiric antibiotics. Consider sputum culture and follow-up chest X-ray in 4-6 weeks.",
        "finding": "Consolidation",
    },
    {
        "keywords": ["nodule", "pulmonary nodule", "lung nodule"],
        "explanation": "Pulmonary nodule detected. Requires monitoring for potential malignancy.",
        "severity": 2.5,
        "recommendation": "Follow-up CT in 3-6 months per Fleischner Society guidelines. Compare with prior imaging if available.",
        "finding": "Pulmonary Nodule",
    },

    # ── MEDIUM SEVERITY ──
    {
        "keywords": ["infection", "opacity", "infiltrate", "pneumonia"],
        "explanation": "Possible pulmonary infection or inflammatory process detected based on opacification patterns.",
        "severity": 2,
        "recommendation": "Clinical correlation recommended. Consider antibiotic therapy and follow-up imaging in 2-4 weeks.",
        "finding": "Infection / Opacity",
    },
    {
        "keywords": ["fracture", "broken", "displaced"],
        "explanation": "Bone fracture identified in imaging.",
        "severity": 2,
        "recommendation": "Orthopedic consultation recommended. Immobilization and pain management advised.",
        "finding": "Fracture",
    },
    {
        "keywords": ["pleural effusion", "effusion", "fluid"],
        "explanation": "Pleural effusion (fluid accumulation) detected in the thoracic cavity.",
        "severity": 2,
        "recommendation": "Monitor progression. Diagnostic thoracentesis may be warranted if symptomatic.",
        "finding": "Pleural Effusion",
    },
    {
        "keywords": ["cardiomegaly", "enlarged heart", "heart enlargement"],
        "explanation": "Cardiomegaly (enlarged heart) detected, suggesting possible cardiac disease.",
        "severity": 2,
        "recommendation": "Echocardiogram recommended. Cardiology consultation advised.",
        "finding": "Cardiomegaly",
    },
    {
        "keywords": ["atelectasis", "collapse"],
        "explanation": "Atelectasis (partial lung collapse) identified.",
        "severity": 2,
        "recommendation": "Respiratory therapy recommended. Consider incentive spirometry and follow-up imaging.",
        "finding": "Atelectasis",
    },
    {
        "keywords": ["edema", "pulmonary edema", "congestion"],
        "explanation": "Pulmonary edema or vascular congestion detected, suggesting fluid overload.",
        "severity": 2,
        "recommendation": "Evaluate for heart failure. Diuretic therapy and cardiology follow-up recommended.",
        "finding": "Pulmonary Edema",
    },
    {
        "keywords": ["herniation", "hernia", "disc herniation"],
        "explanation": "Herniation detected. May cause nerve compression or functional impairment.",
        "severity": 2,
        "recommendation": "Neurosurgery or orthopedic consultation recommended. MRI for further evaluation.",
        "finding": "Herniation",
    },

    # ── LOW-MEDIUM SEVERITY ──
    {
        "keywords": ["mild opacity", "hazy opacity", "ground glass"],
        "explanation": "Mild or ground-glass opacity observed, which may indicate early inflammatory process.",
        "severity": 1.5,
        "recommendation": "Follow-up imaging in 4-6 weeks recommended. Clinical correlation advised.",
        "finding": "Ground-Glass Opacity",
    },
    {
        "keywords": ["calcification", "calcified"],
        "explanation": "Calcification noted, which is often benign but may indicate prior infection or granulomatous disease.",
        "severity": 1.5,
        "recommendation": "Likely benign. Compare with prior imaging. Follow-up only if symptomatic.",
        "finding": "Calcification",
    },
    {
        "keywords": ["scoliosis", "curvature", "spinal curvature"],
        "explanation": "Spinal curvature (scoliosis) identified.",
        "severity": 1.5,
        "recommendation": "Orthopedic evaluation recommended if symptomatic or progressive.",
        "finding": "Scoliosis",
    },
    {
        "keywords": ["degenerative", "arthritis", "osteophyte", "spondylosis"],
        "explanation": "Degenerative changes or arthritis detected — age-related wear.",
        "severity": 1.5,
        "recommendation": "Conservative management with physical therapy. Follow-up as needed.",
        "finding": "Degenerative Changes",
    },

    # ── LOW SEVERITY ──
    {
        "keywords": ["normal", "unremarkable", "no acute", "no active", "within normal limits"],
        "explanation": "No significant abnormalities detected. Imaging findings are within normal limits.",
        "severity": 0.5,
        "recommendation": "No immediate clinical intervention required. Routine follow-up recommended.",
        "finding": "Normal / Unremarkable",
    },
    {
        "keywords": ["stable", "unchanged", "no interval change"],
        "explanation": "Findings are stable compared to prior studies with no interval change.",
        "severity": 0.5,
        "recommendation": "Continue current management. Routine follow-up imaging as scheduled.",
        "finding": "Stable / Unchanged",
    },
]


def _classify_severity(score):
    """Convert numeric severity score to categorical label."""
    if score >= 2.5:
        return "🔴 High"
    elif score >= 1.5:
        return "🟡 Medium"
    else:
        return "🟢 Low"


def interpret_report(full_text):
    """
    Interpret a radiology report using comprehensive rule-based analysis.

    Args:
        full_text (str): The FULL report text (not summary).

    Returns:
        dict with keys: explanation, severity, recommendation, key_findings
    """
    if not full_text or not full_text.strip():
        return {
            "explanation": "No report text provided for interpretation.",
            "severity": "⚪ Unknown",
            "recommendation": "Please provide a valid radiology report for analysis.",
            "key_findings": [],
        }

    text_lower = full_text.lower()
    matched_rules = []

    for rule in MEDICAL_RULES:
        for keyword in rule["keywords"]:
            if keyword in text_lower:
                matched_rules.append(rule)
                break  # Only match each rule once

    if not matched_rules:
        return {
            "explanation": "The report content could not be matched to known medical patterns. "
                           "The text may be non-standard or contain insufficient clinical information.",
            "severity": "⚪ Unknown",
            "recommendation": "Manual review by a radiologist is recommended. "
                               "Please ensure the report contains standard medical terminology.",
            "key_findings": [],
        }

    # Use highest-severity match as the primary finding
    matched_rules.sort(key=lambda r: r["severity"], reverse=True)

    # Build comprehensive explanation from all findings
    explanations = []
    recommendations = []
    key_findings = []
    seen_explanations = set()

    for rule in matched_rules:
        if rule["explanation"] not in seen_explanations:
            explanations.append(f"• {rule['explanation']}")
            seen_explanations.add(rule["explanation"])
        if rule["recommendation"] not in recommendations:
            recommendations.append(rule["recommendation"])
        # Collect key findings with severity level
        key_findings.append({
            "finding": rule.get("finding", "Unknown"),
            "severity_score": rule["severity"],
            "severity_label": _classify_severity(rule["severity"]),
        })

    max_severity = max(r["severity"] for r in matched_rules)

    return {
        "explanation": "\n".join(explanations),
        "severity": _classify_severity(max_severity),
        "recommendation": " ".join(dict.fromkeys(recommendations)),  # deduplicated, ordered
        "key_findings": key_findings,
    }