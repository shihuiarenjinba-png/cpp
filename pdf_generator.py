from fpdf import FPDF
import tempfile
import os

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
        self.multi_cell(0, 5, str(body))
        self.ln()

def create_pdf_report(payload, figs):
    pdf = PDF()
    pdf.add_page()

    # 1. Basic Stats
    if 'stats' in payload:
        pdf.chapter_title("1. Risk & Return Metrics")
        pdf.chapter_body(payload['stats'])

    # 2. Charts (Attempt to add images safely)
    if figs:
        pdf.chapter_title("2. Visual Analysis")
        for title, fig in figs.items():
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                    # Kaleidoを使って画像を保存
                    fig.write_image(tmpfile.name, format="png", engine="kaleido")
                    tmpfile_path = tmpfile.name
                
                pdf.set_font('Arial', 'I', 10)
                pdf.cell(0, 10, f"Chart: {title}", 0, 1)
                pdf.image(tmpfile_path, w=170)
                os.unlink(tmpfile_path) # 一時ファイルを削除
                pdf.ln(5)
            except Exception as e:
                # 画像が失敗してもテキストで記録して続行
                pdf.set_font('Arial', 'I', 8)
                pdf.set_text_color(255, 0, 0)
                pdf.cell(0, 10, f"[Image Error: {title}] - {str(e)}", 0, 1)
                pdf.set_text_color(0, 0, 0)

    # 3. Factor Analysis
    if 'factor_comment' in payload:
        pdf.add_page()
        pdf.chapter_title("3. Factor Analysis & AI Insight")
        pdf.chapter_body(payload['factor_comment'])

    # 4. Monte Carlo
    if 'mc_stats' in payload:
        pdf.chapter_title("4. Future Projections (Monte Carlo)")
        pdf.chapter_body(payload['mc_stats'])

    # 出力 (fpdf2では bytes が直接返されるため、余計なエンコードは不要)
    return pdf.output()
