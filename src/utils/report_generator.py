"""
Report Generator Module
-----------------------
Generates downloadable HTML reports with premium styling.
Supports text analysis, OCR upload, and scan analysis modes.
"""

from datetime import datetime


def _escape_html(text):
    """Escape HTML special characters."""
    if not text:
        return ""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
            .replace("\n", "<br>"))


def generate_report(summary, severity, recommendation, explanation, mode="text", key_findings=None):
    """
    Generate a styled HTML report for download.

    Args:
        summary: Clinical summary text
        severity: Severity level string
        recommendation: Clinical recommendation
        explanation: Detailed interpretation
        mode: Analysis mode ('text', 'report_upload', 'scan')
        key_findings: Optional list of key findings dicts

    Returns:
        str: Complete HTML document as string
    """
    timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")

    mode_labels = {
        "text": "Text Report Analysis",
        "report_upload": "Uploaded Report Analysis (OCR)",
        "scan": "Medical Scan Analysis"
    }
    mode_label = mode_labels.get(mode, "Report Analysis")

    # Severity color mapping
    if "High" in severity:
        sev_color = "#ef4444"
        sev_bg = "rgba(239, 68, 68, 0.1)"
    elif "Medium" in severity:
        sev_color = "#f59e0b"
        sev_bg = "rgba(245, 158, 11, 0.1)"
    elif "Low" in severity:
        sev_color = "#10b981"
        sev_bg = "rgba(16, 185, 129, 0.1)"
    else:
        sev_color = "#6b7280"
        sev_bg = "rgba(107, 114, 128, 0.1)"

    # Build key findings table
    findings_html = ""
    if key_findings:
        rows = ""
        for f in key_findings:
            label = f.get("severity_label", "")
            if "High" in label:
                badge_style = "background: rgba(239,68,68,0.15); color: #fca5a5; border: 1px solid rgba(239,68,68,0.3);"
            elif "Medium" in label:
                badge_style = "background: rgba(245,158,11,0.15); color: #fcd34d; border: 1px solid rgba(245,158,11,0.3);"
            else:
                badge_style = "background: rgba(16,185,129,0.15); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.3);"

            rows += f"""
            <tr>
              <td style="padding: 10px 16px; border-bottom: 1px solid rgba(99,102,241,0.1); color: #e2e8f0;">
                {_escape_html(f.get('finding', 'N/A'))}
              </td>
              <td style="padding: 10px 16px; border-bottom: 1px solid rgba(99,102,241,0.1); text-align: center;">
                <span style="padding: 4px 14px; border-radius: 20px; font-size: 12px; font-weight: 600; {badge_style}">
                  {_escape_html(label)}
                </span>
              </td>
            </tr>"""

        findings_html = f"""
        <div class="section">
          <h2>Key Findings</h2>
          <table style="width: 100%; border-collapse: collapse; margin-top: 8px;">
            <thead>
              <tr>
                <th style="padding: 10px 16px; text-align: left; color: #a5b4fc; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; border-bottom: 1px solid rgba(99,102,241,0.2);">Finding</th>
                <th style="padding: 10px 16px; text-align: center; color: #a5b4fc; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; border-bottom: 1px solid rgba(99,102,241,0.2);">Severity</th>
              </tr>
            </thead>
            <tbody>
              {rows}
            </tbody>
          </table>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Radiology Report — {timestamp}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
  body {{
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #0f172a; color: #e2e8f0;
    padding: 40px; line-height: 1.6;
  }}
  .container {{
    max-width: 800px; margin: 0 auto;
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 16px; padding: 48px;
    box-shadow: 0 25px 50px rgba(0, 0, 0, 0.5);
  }}
  .header {{
    text-align: center; margin-bottom: 40px;
    border-bottom: 1px solid rgba(99, 102, 241, 0.2);
    padding-bottom: 24px;
  }}
  .header h1 {{
    font-size: 28px; font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }}
  .header .meta {{
    color: #94a3b8; font-size: 14px; margin-top: 8px;
  }}
  .badge {{
    display: inline-block; padding: 4px 16px;
    border-radius: 20px; font-size: 13px; font-weight: 600;
    background: rgba(99, 102, 241, 0.15); color: #a5b4fc;
    margin-top: 8px;
  }}
  .section {{
    margin-bottom: 28px; padding: 20px;
    background: rgba(30, 41, 59, 0.5);
    border-radius: 12px;
    border-left: 4px solid #6366f1;
  }}
  .section h2 {{
    font-size: 16px; text-transform: uppercase;
    letter-spacing: 1px; color: #a5b4fc; margin-bottom: 12px;
  }}
  .section p {{ color: #cbd5e1; white-space: pre-line; }}
  .severity-box {{
    padding: 16px 24px; border-radius: 12px;
    border: 1px solid {sev_color}; background: {sev_bg};
    text-align: center; margin-bottom: 28px;
  }}
  .severity-box .level {{
    font-size: 24px; font-weight: 700; color: {sev_color};
  }}
  .footer {{
    text-align: center; margin-top: 32px;
    padding-top: 20px;
    border-top: 1px solid rgba(99, 102, 241, 0.2);
    color: #64748b; font-size: 12px;
  }}
  .disclaimer {{
    margin-top: 24px; padding: 16px;
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-radius: 8px; font-size: 13px; color: #fbbf24;
  }}
  @media print {{
    body {{ background: white; color: #1e293b; padding: 20px; }}
    .container {{ box-shadow: none; border: 1px solid #e2e8f0; }}
    .section {{ border-left-color: #6366f1; }}
    .header h1 {{ -webkit-text-fill-color: #6366f1; }}
  }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>🏥 AI Radiology Report</h1>
    <div class="meta">{timestamp}</div>
    <span class="badge">{mode_label}</span>
  </div>

  <div class="section">
    <h2>🧠 Clinical Summary</h2>
    <p>{_escape_html(summary)}</p>
  </div>

  <div class="section">
    <h2>📋 Clinical Interpretation</h2>
    <p>{_escape_html(explanation)}</p>
  </div>

  {findings_html}

  <div class="severity-box">
    <div style="color: #94a3b8; font-size: 13px; text-transform: uppercase; letter-spacing: 1px;">Severity Level</div>
    <div class="level">{_escape_html(severity)}</div>
  </div>

  <div class="section">
    <h2>💡 Recommendation</h2>
    <p>{_escape_html(recommendation)}</p>
  </div>

  <div class="disclaimer">
    ⚠️ This report is generated by an AI system and is intended for informational purposes only.
    It should not replace professional medical diagnosis. Always consult a qualified radiologist
    or physician for clinical decisions.
  </div>

  <div class="footer">
    <p>Generated by AI Radiology Assistant</p>
    <p>For accurate detection, deep learning models like U-Net trained on
    NIH Chest X-ray or CheXpert datasets can be used.</p>
  </div>
</div>
</body>
</html>"""

    return html