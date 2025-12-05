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
    ("LT011", "Brief", "A written statement submitted by the lawyer for each side in a case that explains the law and facts supporting that party's case", "US Courts"),
    ("LT012", "Burden of Proof", "The duty to prove disputed facts in a case resting on the party asserting the affirmative of an issue", "DOJ"),
    ("LT013", "Case Law", "Law established by previous decisions of appellate courts particularly the Supreme Court", "US Courts"),
    ("LT014", "Chambers", "The offices of a judge and the judge's staff", "US Courts"),
    ("LT015", "Change of Venue", "Moving a lawsuit or criminal trial to another place for trial", "DOJ"),
    ("LT016", "Civil Action", "A lawsuit brought to enforce protect or redress private or civil rights", "US Courts"),
    ("LT017", "Class Action", "A lawsuit in which one or more members of a large group represent the entire group", "US Courts"),
    ("LT018", "Clerk of Court", "An officer appointed by the court to work with the chief judge in overseeing the court's administration", "US Courts"),
    ("LT019", "Common Law", "The legal system that originated in England and is now in use in the United States based on judicial decisions", "DOJ"),
    ("LT020", "Complaint", "A written statement that begins a civil lawsuit containing the plaintiff's allegations against the defendant", "US Courts"),
    ("LT021", "Concurrent Sentence", "Prison terms for two or more offenses to be served at the same time rather than consecutively", "DOJ"),
    ("LT022", "Consecutive Sentence", "Prison terms for two or more offenses to be served one after the other", "DOJ"),
    ("LT023", "Contract", "An agreement between two or more persons that creates an obligation to do or not to do a particular thing", "US Courts"),
    ("LT024", "Conviction", "A judgment of guilt against a criminal defendant", "US Courts"),
    ("LT025", "Court Reporter", "A person who makes a word-for-word record of what is said in court and produces a transcript upon request", "US Courts"),
    ("LT026", "Creditor", "A person to whom money is owed by another person called the debtor", "US Courts"),
    ("LT027", "Cross-Examination", "Questioning of a witness by the opposing party during a trial or deposition", "DOJ"),
    ("LT028", "Damages", "Money paid by defendants to successful plaintiffs in civil cases to compensate the plaintiffs for their injuries", "US Courts"),
    ("LT029", "Debtor", "A person who owes money to a creditor", "US Courts"),
    ("LT030", "Declaratory Judgment", "A judge's statement about someone's rights without ordering that anything be done", "US Courts"),
    ("LT031", "Default Judgment", "A judgment rendered because of the defendant's failure to answer or appear in court", "DOJ"),
    ("LT032", "Defendant", "The person or entity defending against a lawsuit or criminal prosecution", "US Courts"),
    ("LT033", "Deposition", "An oral statement made before an officer authorized by law to administer oaths taken before trial", "US Courts"),
    ("LT034", "Discovery", "The pretrial process by which parties gather information from each other", "US Courts"),
    ("LT035", "Dismissal", "An order or judgment disposing of a case without a trial", "DOJ"),
    ("LT036", "Docket", "A log containing the complete history of each case in the form of brief chronological entries", "US Courts"),
    ("LT037", "Due Process", "The right to a fair and impartial hearing including notice of charges and opportunity to be heard", "DOJ"),
    ("LT038", "En Banc", "A proceeding in which the entire court participates in the decision rather than a subset of judges", "US Courts"),
    ("LT039", "Evidence", "Information presented in testimony or in documents used to persuade the fact finder", "US Courts"),
    ("LT040", "Executor", "A personal representative named in a will to administer the estate of a deceased person", "US Courts"),
    ("LT041", "Exempt Assets", "Property that a debtor is allowed to retain free from the claims of creditors", "US Courts"),
    ("LT042", "Expert Witness", "A witness with specialized knowledge or training who testifies about professional or scientific matters", "DOJ"),
    ("LT043", "Felony", "A serious crime usually punishable by imprisonment for more than one year or death", "US Courts"),
    ("LT044", "Filing", "The act of submitting a document to the court clerk for inclusion in the case file", "US Courts"),
    ("LT045", "Foreclosure", "A legal proceeding to terminate a property owner's rights in property", "DOJ"),
    ("LT046", "Fraudulent Transfer", "A transfer of a debtor's property made with intent to defraud creditors", "US Courts"),
    ("LT047", "Grand Jury", "A body of citizens who listen to evidence and determine whether there is probable cause to believe a crime was committed", "US Courts"),
    ("LT048", "Habeas Corpus", "A writ that is usually used to bring a prisoner before the court to determine the legality of imprisonment", "US Courts"),
    ("LT049", "Hearsay", "Evidence presented by a witness based on what others have said rather than personal knowledge", "DOJ"),
    ("LT050", "Impeachment", "The process of calling something into question such as the testimony of a witness", "US Courts"),
    ("LT051", "In Camera", "In chambers or in private where the public and jury are excluded", "DOJ"),
    ("LT052", "Indictment", "The formal charge issued by a grand jury stating there is enough evidence to require a trial", "US Courts"),
    ("LT053", "Injunction", "A court order preventing one or more named parties from taking some action", "US Courts"),
    ("LT054", "Interrogatories", "Written questions developed by one party and sent to another party to be answered under oath", "DOJ"),
    ("LT055", "Joint Administration", "A court-approved mechanism under which two or more cases can be administered together", "US Courts"),
    ("LT056", "Judgment", "The official decision of a court finally determining the respective rights and claims of the parties", "US Courts"),
    ("LT057", "Jurisdiction", "The legal authority of a court to hear and decide a case", "US Courts"),
    ("LT058", "Jury", "A group of citizens who hear the evidence presented by both sides at trial and determine the facts in dispute", "US Courts"),
    ("LT059", "Jury Instructions", "A judge's directions to the jury before it begins deliberations regarding the law that applies to the case", "DOJ"),
    ("LT060", "Lien", "A charge or claim on property as security for a debt or obligation", "US Courts"),
    ("LT061", "Litigation", "A case lawsuit or action in a civil or criminal court", "US Courts"),
    ("LT062", "Magistrate Judge", "A judicial officer who assists district court judges in preparing cases for trial", "US Courts"),
    ("LT063", "Malpractice", "Professional misconduct or unreasonable lack of skill by a professional", "DOJ"),
    ("LT064", "Mandamus", "A writ issued by a court ordering a public official to perform an act", "US Courts"),
    ("LT065", "Material Witness", "A witness whose testimony is essential to a case", "DOJ"),
    ("LT066", "Mediation", "A form of alternative dispute resolution in which a neutral third party assists in resolving a dispute", "US Courts"),
    ("LT067", "Misdemeanor", "A crime less serious than a felony usually punishable by fine or imprisonment for less than one year", "US Courts"),
    ("LT068", "Mistrial", "An invalid trial caused by fundamental error resulting in termination of trial before a verdict", "DOJ"),
    ("LT069", "Motion", "A request by a litigant asking the court to decide an issue in the case", "US Courts"),
    ("LT070", "Motion in Limine", "A pretrial motion requesting the court to prohibit the opposing party from referring to certain matters", "DOJ"),
    ("LT071", "Negligence", "Failure to exercise the degree of care that a reasonable person would exercise under the same circumstances", "US Courts"),
    ("LT072", "Nolo Contendere", "No contest - a plea by which a defendant does not expressly admit guilt", "DOJ"),
    ("LT073", "Notary Public", "A public officer authorized to administer oaths and certify documents", "US Courts"),
    ("LT074", "Oath", "A solemn pledge made under a sense of responsibility in attestation of truth of a statement", "DOJ"),
    ("LT075", "Objection", "A protest by an attorney challenging a statement or question made at trial", "US Courts"),
    ("LT076", "Opinion", "A judge's written explanation of a decision or judgment", "US Courts"),
    ("LT077", "Oral Argument", "An opportunity for lawyers to summarize their position before the court and answer questions", "US Courts"),
    ("LT078", "Order", "A written command or direction issued by a judge", "DOJ"),
    ("LT079", "Ordinance", "A law passed by a local government such as a city or county", "US Courts"),
    ("LT080", "Parole", "The supervised release of a prisoner before completion of the full sentence", "DOJ"),
    ("LT081", "Party", "A person business or government agency actively involved in the prosecution or defense of a legal proceeding", "US Courts"),
    ("LT082", "Perjury", "The criminal offense of making false statements under oath", "DOJ"),
    ("LT083", "Petition", "A formal written request asking the court for specific relief", "US Courts"),
    ("LT084", "Petitioner", "The party filing a petition especially an appeal", "US Courts"),
    ("LT085", "Plaintiff", "The person who files the complaint in a civil lawsuit", "US Courts"),
    ("LT086", "Plea", "In a criminal proceeding the defendant's declaration in open court of guilt or innocence", "DOJ"),
    ("LT087", "Plea Bargain", "An agreement between the prosecutor and defendant where the defendant pleads guilty in exchange for concessions", "US Courts"),
    ("LT088", "Pleadings", "Written statements of the parties in a civil case detailing their claims defenses and replies", "US Courts"),
    ("LT089", "Precedent", "A court decision in an earlier case with facts and law similar to a dispute currently before a court", "US Courts"),
    ("LT090", "Preliminary Injunction", "A court order to preserve the status quo until a hearing can be held", "DOJ"),
    ("LT091", "Preponderance of Evidence", "The greater weight of the evidence required in most civil cases", "US Courts"),
    ("LT092", "Presentence Report", "A report prepared by a probation officer after conviction to assist the judge in sentencing", "DOJ"),
    ("LT093", "Pretrial Conference", "A meeting between the judge and lawyers to narrow issues prepare for trial and explore settlement", "US Courts"),
    ("LT094", "Probation", "A sentencing alternative allowing a convicted defendant to serve their sentence outside of jail under supervision", "US Courts"),
    ("LT095", "Pro Se", "Representing oneself without an attorney in a legal proceeding", "US Courts"),
    ("LT096", "Prosecutor", "A government attorney who presents the state's case against a defendant in a criminal proceeding", "DOJ"),
    ("LT097", "Quash", "To nullify void or declare invalid such as quashing a subpoena", "US Courts"),
    ("LT098", "Reasonable Doubt", "The level of certainty a juror must have to find a defendant guilty of a crime", "DOJ"),
    ("LT099", "Rebuttal", "Evidence offered to contradict or disprove evidence presented by the opposing party", "US Courts"),
    ("LT100", "Record", "A written account of all the proceedings in a case including all pleadings evidence and exhibits", "US Courts"),
    ("LT101", "Remand", "When an appellate court sends a case back to a lower court for further proceedings", "US Courts"),
    ("LT102", "Respondent", "The party against whom an appeal or motion is filed", "DOJ"),
    ("LT103", "Restitution", "The act of compensating someone for loss damage or injury by the party responsible", "US Courts"),
    ("LT104", "Retainer", "An advance payment to an attorney for legal services to be rendered", "DOJ"),
    ("LT105", "Sanction", "A penalty or other enforcement measure used to provide incentives for compliance with the law", "US Courts"),
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
