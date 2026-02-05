from fpdf import FPDF
import tempfile
import os
import re

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
        # タイトルからも特殊文字を除去
        safe_title = self.clean_text(str(title))
        self.cell(0, 6, safe_title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        # 本文から特殊文字を除去して書き込む
        safe_body = self.clean_text(str(body))
        self.multi_cell(0, 5, safe_body)
        self.ln()

    def clean_text(self, text):
        """
        絵文字や特殊記号を取り除き、標準的なフォントで表示可能な文字のみにする関数
        """
        # Latin-1（標準フォントの範囲）に変換できない文字を無視して削除
        return text.encode('ascii', 'ignore').decode('ascii')

def create_pdf_report(payload, figs):
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    if not payload:
        pdf.chapter_title("No Data Available")
    else:
        # 1. Stats
        if 'stats' in payload:
            pdf.chapter_title("1. Risk & Return Metrics")
            pdf.chapter_body(payload['stats'])

        # 2. Charts
        if figs:
            pdf.chapter_title("2. Visual Analysis")
            for title, fig in figs.items():
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                        fig.write_image(tmpfile.name, format="png", engine="kaleido")
                        img_path = tmpfile.name
                    
                    pdf.set_font('Arial', 'I', 10)
                    pdf.cell(0, 8, f"Chart: {pdf.clean_text(title)}", 0, 1)
                    pdf.image(img_path, w=170)
                    os.unlink(img_path)
                    pdf.ln(5)
                except:
                    pdf.cell(0, 10, "[Chart Render Skipped]", 0, 1)

        # 3. Factor Analysis
        if 'factor_comment' in payload:
            pdf.add_page()
            pdf.chapter_title("3. Factor Analysis & AI Insight")
            pdf.chapter_body(payload['factor_comment'])

    # fpdf2のoutput()はbytearrayを返すため、Streamlit用にbytesに変換する
    return bytes(pdf.output())
