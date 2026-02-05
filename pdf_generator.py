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
        self.cell(0, 6, str(title), 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        # latin-1変換を完全に削除し、文字列として安全に処理
        self.multi_cell(0, 5, str(body))
        self.ln()

def create_pdf_report(payload, figs):
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # 1. Check if payload has data
    if not payload:
        pdf.chapter_title("No Data Available")
        pdf.chapter_body("Please run the simulation to generate data.")
    else:
        # 📊 Stats Section
        if 'stats' in payload:
            pdf.chapter_title("1. Risk & Return Metrics")
            pdf.chapter_body(payload['stats'])

        # 📈 Charts Section (Safe Image Generation)
        if figs:
            pdf.chapter_title("2. Visual Analysis")
            for title, fig in figs.items():
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                        # Kaleidoで画像を保存
                        fig.write_image(tmpfile.name, format="png", engine="kaleido")
                        img_path = tmpfile.name
                    
                    pdf.set_font('Arial', 'I', 10)
                    pdf.cell(0, 8, f"Chart: {title}", 0, 1)
                    pdf.image(img_path, w=170)
                    os.unlink(img_path) # 一時ファイルを削除
                    pdf.ln(5)
                except Exception as e:
                    pdf.set_text_color(200, 0, 0)
                    pdf.cell(0, 10, f"[Image Render Error: {title}]", 0, 1)
                    pdf.set_text_color(0, 0, 0)

        # 🧠 Analysis Section
        if 'factor_comment' in payload:
            pdf.add_page()
            pdf.chapter_title("3. Factor Analysis & AI Insight")
            pdf.chapter_body(payload['factor_comment'])

    # 【重要】fpdf2では引数なしのoutput()が bytes を返します
    return pdf.output()
