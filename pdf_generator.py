from fpdf import FPDF
import tempfile
import os
import re

class PDF(FPDF):
    def __init__(self):
        super().__init__()
        # 英語標準フォントを使用（絵文字エラー回避のためArial/Helvetica）
        self.set_font('Arial', '', 10)

    def clean_text(self, text):
        """絵文字や特殊文字を除去する関数"""
        if not text: return ""
        text = str(text)
        return re.sub(r'[^\x00-\x7F]+', '', text)

    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Portfolio Analysis Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()} | Statistical Analysis Report', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(240, 240, 240)
        self.cell(0, 8, self.clean_text(title), 0, 1, 'L', 1)
        self.ln(3)

    def draw_table(self, data_dict):
        """
        辞書データを綺麗な表形式で出力する関数
        """
        self.set_font('Arial', 'B', 10)
        self.set_fill_color(200, 220, 255)
        
        # ヘッダー
        self.cell(90, 8, " Metric", 1, 0, 'L', 1)
        self.cell(80, 8, " Value", 1, 1, 'L', 1)
        
        self.set_font('Arial', '', 10)
        for key, value in data_dict.items():
            self.check_space(8)
            # キーと値を綺麗に配置
            self.cell(90, 8, f" {self.clean_text(key)}", 1, 0, 'L')
            self.cell(80, 8, f" {self.clean_text(value)}", 1, 1, 'L')
        self.ln(5)

    def check_space(self, height_needed):
        if self.get_y() + height_needed > 275:
            self.add_page()

def create_pdf_report(payload, figs):
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    if not payload:
        pdf.chapter_title("No Data Available")
        return bytes(pdf.output())

    # 1. Advisor's Note (サイドバーコメント)
    if 'advisor_note' in payload and payload['advisor_note']:
        pdf.chapter_title("1. Advisor's Note")
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 5, pdf.clean_text(payload['advisor_note']))
        pdf.ln(5)

    # 2. Executive Summary (AI Diagnosis)
    if 'ai_diagnosis' in payload:
        pdf.check_space(40)
        pdf.chapter_title("2. Executive Summary (AI Diagnosis)")
        diag = payload['ai_diagnosis']
        
        # 診断結果を表形式に
        diag_data = {
            "Portfolio Type": diag.get('status', '-'),
            "Risk Assessment": diag.get('risk', '-'),
            "Recommended Action": diag.get('action', '-')
        }
        pdf.draw_table(diag_data)

        if 'detailed_review' in payload:
             pdf.set_font('Arial', 'B', 10)
             pdf.cell(0, 6, "Detailed AI Analysis:", 0, 1)
             pdf.set_font('Arial', '', 9)
             pdf.multi_cell(0, 5, pdf.clean_text(payload['detailed_review']))
             pdf.ln(5)

    # 3. Key Risk & Return Metrics (表形式)
    if 'metrics' in payload:
        pdf.check_space(40)
        pdf.chapter_title("3. Key Performance Metrics")
        pdf.draw_table(payload['metrics'])

    # 4. Visual Analysis (Graphs)
    if figs:
        pdf.add_page()
        pdf.chapter_title("4. Visual Analysis")
        for key in ['pie', 'history', 'mc', 'correlation', 'factor_beta', 'attribution']:
            if key in figs:
                try:
                    pdf.check_space(100)
                    pdf.set_font('Arial', 'B', 10)
                    pdf.cell(0, 8, f"Chart: {key.upper()}", 0, 1)
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                        figs[key].write_image(tmpfile.name, format="png", engine="kaleido", scale=2)
                        pdf.image(tmpfile.name, w=170)
                        os.unlink(tmpfile.name)
                    pdf.ln(10)
                except:
                    pass

    # 5. Disclaimer (免責事項 - 責任転嫁の文言)
    pdf.add_page()
    pdf.chapter_title("5. Risk Disclosure & Disclaimer")
    disclaimer_text = (
        "This report is provided for informational purposes only and does not constitute financial, investment, or legal advice. "
        "The analysis is based on historical market data and statistical simulations (such as Monte Carlo analysis). "
        "Please note that historical performance is not an indicator of future results.\n\n"
        "Investing in financial markets involves risk, including the potential loss of principal. "
        "The projections provided in this report are probabilistic in nature and do not guarantee any specific profit or outcome. "
        "The authors and the system shall not be held liable for any financial losses or damages resulting from decisions made based on this report. "
        "It is strongly recommended to consult with a certified financial advisor before making any investment decisions."
    )
    pdf.set_font('Arial', 'I', 9)
    pdf.set_text_color(100, 100, 100) # グレー色で少し控えめに
    pdf.multi_cell(0, 5, disclaimer_text)
    
    return bytes(pdf.output())
