from fpdf import FPDF
import tempfile
import os
import re

class PDF(FPDF):
    def __init__(self):
        super().__init__()
        # 以前お伝えしたフォント設定（ファイルがない場合はArialにフォールバック）
        self.font_path = 'ipaexg.ttf' 
        self.use_custom_font = os.path.exists(self.font_path)
        
        if self.use_custom_font:
            self.add_font('CustomFont', '', self.font_path, uni=True)
            self.set_font('CustomFont', '', 10)
        else:
            self.set_font('Arial', '', 10)

    def clean_text(self, text):
        """
        エラーの原因となる絵文字や特殊記号を、PDF出力前に取り除く関数
        """
        if not text:
            return ""
        # 1. 文字列に変換
        text = str(text)
        # 2. 絵文字などの非ASCII文字を除去（英語の文字と標準記号のみ残す）
        # これにより、'⚠ Alert' は ' Alert' になり、エラーを防ぎます
        return re.sub(r'[^\x00-\x7F]+', '', text)

    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'Portfolio Analysis Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(230, 230, 230)
        # タイトルから絵文字を消去
        safe_title = self.clean_text(title)
        self.cell(0, 8, safe_title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        # 本文から絵文字を消去
        safe_body = self.clean_text(body)
        self.multi_cell(0, 5, safe_body)
        self.ln()

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

    # --- 各セクションで clean_text を通して出力 ---
    
    # 1. Advisor's Note
    if 'advisor_note' in payload and payload['advisor_note']:
        pdf.chapter_title("1. Advisor's Note")
        pdf.chapter_body(payload['advisor_note'])

    # 2. AI Diagnosis
    if 'ai_diagnosis' in payload:
        diag = payload['ai_diagnosis']
        pdf.check_space(40)
        pdf.chapter_title("2. Executive Summary (AI Diagnosis)")
        
        summary_text = (
            f"Type: {diag.get('status', '-')}\n"
            f"Risk Assessment: {diag.get('risk', '-')}\n"
            f"Action Plan: {diag.get('action', '-')}\n"
        )
        pdf.chapter_body(summary_text)

        if 'detailed_review' in payload:
             pdf.chapter_body("Detailed Analysis:\n" + str(payload['detailed_review']))

    # 3. Metrics
    if 'metrics' in payload:
        pdf.check_space(40)
        pdf.chapter_title("3. Key Metrics")
        metrics_str = "\n".join([f"- {k}: {v}" for k, v in payload['metrics'].items()])
        pdf.chapter_body(metrics_str)

    # 4. Visual Analysis (Graphs)
    if figs:
        pdf.add_page()
        pdf.chapter_title("4. Visual Analysis")
        for key in ['pie', 'history', 'mc', 'correlation', 'factor_beta', 'attribution']:
            if key in figs:
                try:
                    pdf.check_space(95)
                    pdf.set_font('Arial', 'B', 10)
                    pdf.cell(0, 8, f"[Chart: {key.upper()}]", 0, 1)

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                        figs[key].write_image(tmpfile.name, format="png", engine="kaleido", scale=2)
                        pdf.image(tmpfile.name, w=170)
                        os.unlink(tmpfile.name)
                    pdf.ln(5)
                except:
                    pdf.cell(0, 10, f"[Chart {key} skipped]", 0, 1)

    return bytes(pdf.output())
