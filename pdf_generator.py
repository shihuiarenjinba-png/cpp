from fpdf import FPDF
import tempfile
import os

# ==========================================
# 📄 Custom PDF Class
# ==========================================
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Portfolio Analysis Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 6, title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        # Check for non-latin characters support issues roughly
        safe_body = body.encode('latin-1', 'replace').decode('latin-1')
        self.multi_cell(0, 5, safe_body)
        self.ln()
    
    def check_page_break(self, height_needed):
        """
        Check if the current page has enough space.
        """
        current_y = self.get_y()
        page_height_limit = 270 
        
        if current_y + height_needed > page_height_limit:
            self.add_page()

# ==========================================
# 🚀 Generator Function
# ==========================================
def create_pdf_report(payload, figs):
    """
    Generate a PDF report and return the binary data.
    """
    try:
        pdf = PDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # --- 0. Advisor's Note ---
        if 'advisor_note' in payload and payload['advisor_note']:
            pdf.set_font('Arial', 'B', 11)
            pdf.cell(0, 10, "Advisor's Note:", 0, 1)
            pdf.set_font('Arial', 'I', 10)
            safe_note = payload['advisor_note'].encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 5, safe_note, border=1)
            pdf.ln(5)

        # --- 1. Executive Summary ---
        pdf.chapter_title("1. Portfolio Executive Summary")
        
        metrics = payload.get('metrics', {})
        pdf.set_font('Arial', '', 10)
        metrics_text = (
            f"CAGR (Growth):      {metrics.get('CAGR', 'N/A')}\n"
            f"Volatility (Risk):  {metrics.get('Volatility', 'N/A')}\n"
            f"Sharpe Ratio:       {metrics.get('Sharpe Ratio', 'N/A')}\n"
            f"Max Drawdown:       {metrics.get('Max Drawdown', 'N/A')}\n"
        )
        pdf.multi_cell(0, 5, metrics_text)
        pdf.ln(3)

        # Detailed AI Review
        if 'detailed_review' in payload:
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(0, 6, "AI Strategic Assessment:", 0, 1)
            pdf.set_font('Arial', '', 10)
            safe_review = payload['detailed_review'].encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 5, safe_review)
            pdf.ln(5)

        # --- AI Diagnosis ---
        if 'ai_diagnosis' in payload:
            diag = payload['ai_diagnosis']
            pdf.check_page_break(50)
            
            pdf.set_font('Arial', 'B', 11)
            pdf.cell(0, 8, "Diagnosis & Action Plan:", 0, 1)

            # Status
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(0, 5, "[ Diagnosis ]", 0, 1)
            pdf.set_font('Arial', '', 10)
            pdf.multi_cell(0, 5, diag['status'].encode('latin-1', 'replace').decode('latin-1'))
            pdf.ln(2)

            # Risk
            pdf.set_font('Arial', 'B', 10)
            pdf.set_text_color(200, 50, 50) 
            pdf.cell(0, 5, "[ Risk Alert ]", 0, 1)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font('Arial', '', 10)
            pdf.multi_cell(0, 5, diag['risk'].encode('latin-1', 'replace').decode('latin-1'))
            pdf.ln(2)

            # Action
            pdf.set_font('Arial', 'B', 10)
            pdf.set_text_color(0, 100, 0)
            pdf.cell(0, 5, "[ Action Plan ]", 0, 1)
            pdf.set_text_color(0, 0, 0) 
            pdf.set_font('Arial', '', 10)
            pdf.multi_cell(0, 5, diag['action'].encode('latin-1', 'replace').decode('latin-1'))
            pdf.ln(5)

        # --- 2. Visual Analysis ---
        pdf.add_page()
        pdf.chapter_title("2. Visual Analysis")
        
        target_order = ['pie', 'correlation', 'history', 'factor_beta', 'mc']
        
        # Temporary directory for images
        with tempfile.TemporaryDirectory() as tmpdirname:
            for key in target_order:
                if key in figs and figs[key]:
                    try:
                        pdf.check_page_break(90)
                        
                        img_path = os.path.join(tmpdirname, f"{key}.png")
                        # Kaleido is used here. If it fails, we skip the image but not the PDF.
                        figs[key].write_image(img_path, width=600, height=350, engine="kaleido")
                        
                        pdf.set_font('Arial', 'B', 10)
                        pdf.cell(0, 8, f"Figure: {key.upper().replace('_', ' ')}", 0, 1)
                        pdf.image(img_path, w=170)
                        pdf.ln(5)
                    except Exception as img_err:
                        print(f"Image Generation Error for {key}: {img_err}")
                        pdf.set_font('Arial', 'I', 8)
                        pdf.set_text_color(255, 0, 0)
                        pdf.cell(0, 5, f"[Image Generation Failed: {key}]", 0, 1)
                        pdf.set_text_color(0, 0, 0)

        # --- 3. Factor Analysis ---
        if 'factor_comment' in payload:
            pdf.check_page_break(40)
            pdf.chapter_title("3. Factor Analysis & AI Insight")
            pdf.chapter_body(payload['factor_comment'])

        # --- 4. Monte Carlo ---
        if 'mc_stats' in payload:
            pdf.check_page_break(30)
            pdf.chapter_title("4. Future Projections (Monte Carlo)")
            pdf.chapter_body(payload['mc_stats'])

        # --- Disclaimer ---
        pdf.ln(10)
        pdf.set_font('Arial', 'I', 8)
        pdf.set_text_color(100, 100, 100)
        disclaimer = (
            "DISCLAIMER: These results are statistical estimates based on historical data "
            "and do not guarantee future performance. Market conditions vary and past performance "
            "is not indicative of future results."
        )
        pdf.multi_cell(0, 4, disclaimer, align='C')

        return pdf.output(dest='S').encode('latin-1')

    except Exception as e:
        print(f"PDF Gen Error: {e}")
        return None
