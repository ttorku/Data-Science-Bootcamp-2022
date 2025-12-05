#!/usr/bin/env python3
"""
Legal Terms Library Generator for Claude Code
Run: pip install openpyxl requests && python create_legal_library.py
"""

import os
import requests
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

# Create directories
os.makedirs("source_documents", exist_ok=True)

# Legal terms data
legal_terms = [
    ("LT001", "Acquittal", "A jury verdict that a criminal defendant is not guilty or the finding of a judge that the evidence is insufficient to support a conviction", "US Courts"),
    ("LT002", "Affidavit", "A written statement of facts confirmed by the oath of the party making it", "DOJ"),
    ("LT003", "Affirmed", "Judgment by appellate courts where the decree or order is declared valid and will stand as decided in the lower court", "DOJ"),
    ("LT004", "Appeal", "A request made after a trial by a party that has lost on one or more issues that a higher court review the decision to determine if it was correct", "US Courts"),
    ("LT005", "Appellant", "The party who appeals a district court's decision usually seeking reversal of that decision", "US Courts"),
    ("LT006", "Appellee", "The party who opposes an appellant's appeal and who seeks to persuade the appeals court to affirm the district court's decision", "US Courts"),
    ("LT007", "Arraignment", "A proceeding in which a criminal defendant is brought into court told of the charges and asked to plead guilty or not guilty", "US Courts"),
    ("LT008", "Bail", "Security given for the release of a criminal defendant or witness from legal custody to secure appearance on the day and time appointed", "DOJ"),
    ("LT009", "Bankruptcy", "A legal procedure for dealing with debt problems of individuals and businesses under title 11 of the United States Code", "US Courts"),
    ("LT010", "Bench Trial", "A trial without a jury in which the judge serves as the fact-finder", "US Courts"),
    # ... Add all 150 terms here (I'll provide the full list)
]

# Create Excel workbook
wb = Workbook()
ws = wb.active
ws.title = "Legal Terms Library"

# Styles
header_font = Font(bold=True, color="FFFFFF", size=12)
header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
border = Border(left=Side(style='thin'), right=Side(style='thin'), top=Side(style='thin'), bottom=Side(style='thin'))

# Headers
headers = ["Legal_ID", "Legal_Term", "Definition", "Source"]
for col, header in enumerate(headers, 1):
    cell = ws.cell(row=1, column=col, value=header)
    cell.font = header_font
    cell.fill = header_fill
    cell.alignment = Alignment(horizontal='center', vertical='center')
    cell.border = border

# Data
for row_idx, (legal_id, term, definition, source) in enumerate(legal_terms, 2):
    ws.cell(row=row_idx, column=1, value=legal_id).border = border
    ws.cell(row=row_idx, column=2, value=term).border = border
    ws.cell(row=row_idx, column=3, value=definition).border = border
    ws.cell(row=row_idx, column=4, value=source).border = border

# Column widths
ws.column_dimensions['A'].width = 12
ws.column_dimensions['B'].width = 30
ws.column_dimensions['C'].width = 100
ws.column_dimensions['D'].width = 15
ws.freeze_panes = 'A2'

wb.save("legal_terms_library.xlsx")
print("✅ Created: legal_terms_library.xlsx")

# Download source documents
sources = [
    ("01_US_Courts_Glossary.html", "https://www.uscourts.gov/glossary"),
    ("02_DOJ_Legal_Terms.html", "https://www.justice.gov/usao/justice-101/glossary"),
    ("03_California_Court_Glossary.pdf", "https://www.saccourt.ca.gov/general/legal-glossaries/docs/english-legal-glossary.pdf"),
    ("04_NY_Courts_Glossary.pdf", "https://www.nycourts.gov/legacyPDFs/courts/6jd/forms/SRForms/glossary_common_legal.pdf"),
    ("05_Indiana_Courts_Glossary.html", "https://www.in.gov/courts/about/glossary/"),
]

headers = {"User-Agent": "Mozilla/5.0"}
print("\nDownloading source documents...")
for name, url in sources:
    try:
        r = requests.get(url, headers=headers, timeout=30)
        mode = "wb" if name.endswith(".pdf") else "w"
        enc = None if name.endswith(".pdf") else "utf-8"
        with open(f"source_documents/{name}", mode, encoding=enc) as f:
            f.write(r.content if name.endswith(".pdf") else r.text)
        print(f"✅ {name}")
    except Exception as e:
        print(f"❌ {name}: {e}")

print("\n🎉 Done! Files ready for legal terms lookup tool.")
