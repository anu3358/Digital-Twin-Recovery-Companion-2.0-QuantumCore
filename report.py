from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

def generate_report(patient_name: str, metrics: dict) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph(f"Recovery Report - {patient_name}", styles['Title']))
    story.append(Spacer(1, 12))
    for k,v in metrics.items():
        story.append(Paragraph(f"<b>{k}</b>: {v}", styles['Normal']))
        story.append(Spacer(1, 8))
    doc.build(story)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf
