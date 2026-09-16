import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.utils import get_column_letter

INPUT_FILE = "golden_candidates.csv"
OUTPUT_FILE = "golden_review.xlsx"

print("Reading golden candidates...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df)}")

# ---------------------------------------------------------
# Create review columns
# ---------------------------------------------------------

# Keep prelabels, but make it clear they are suggestions.
df["final_intent"] = ""
df["intent_confidence"] = ""
df["should_escalate"] = ""
df["escalation_reason_final"] = ""
df["expected_resolution_final"] = ""
df["review_notes_final"] = ""

# Put the important columns first.
columns = [
    "case_id",
    "customer_text",
    "historical_verizon_reply",

    "prelabel_intent",
    "prelabel_confidence",

    "final_intent",
    "intent_confidence",
    "should_escalate",
    "escalation_reason_final",
    "expected_resolution_final",
    "review_notes_final",

    "customer_tweet_id",
    "customer_created_at",
    "support_tweet_id",
    "support_created_at",
]

df = df[columns]

# ---------------------------------------------------------
# Write Excel
# ---------------------------------------------------------

print("Creating Excel workbook...")

df.to_excel(
    OUTPUT_FILE,
    index=False,
    sheet_name="Golden Review"
)

wb = load_workbook(OUTPUT_FILE)

ws = wb["Golden Review"]

# ---------------------------------------------------------
# Header formatting
# ---------------------------------------------------------

for cell in ws[1]:
    cell.font = Font(bold=True)
    cell.alignment = Alignment(
        horizontal="center",
        vertical="center",
        wrap_text=True
    )

# Freeze header.
ws.freeze_panes = "A2"

# Add autofilter.
ws.auto_filter.ref = ws.dimensions

# ---------------------------------------------------------
# Dropdown values
# ---------------------------------------------------------

intent_values = (
    "I01,I02,I03,I04,I05,I06,"
    "I07,I08,I09,I10,I11,I12"
)

confidence_values = "High,Medium,Low"

escalation_values = "Yes,No"

intent_validation = DataValidation(
    type="list",
    formula1=f'"{intent_values}"',
    allow_blank=True
)

confidence_validation = DataValidation(
    type="list",
    formula1=f'"{confidence_values}"',
    allow_blank=True
)

escalation_validation = DataValidation(
    type="list",
    formula1=f'"{escalation_values}"',
    allow_blank=True
)

ws.add_data_validation(intent_validation)
ws.add_data_validation(confidence_validation)
ws.add_data_validation(escalation_validation)

# Find column numbers.
headers = {
    cell.value: cell.column
    for cell in ws[1]
}

final_intent_col = headers["final_intent"]
confidence_col = headers["intent_confidence"]
escalation_col = headers["should_escalate"]

# Apply dropdowns to all 200 rows.
intent_validation.add(
    f"{get_column_letter(final_intent_col)}2:"
    f"{get_column_letter(final_intent_col)}{len(df)+1}"
)

confidence_validation.add(
    f"{get_column_letter(confidence_col)}2:"
    f"{get_column_letter(confidence_col)}{len(df)+1}"
)

escalation_validation.add(
    f"{get_column_letter(escalation_col)}2:"
    f"{get_column_letter(escalation_col)}{len(df)+1}"
)

# ---------------------------------------------------------
# Column widths
# ---------------------------------------------------------

widths = {
    "A": 12,   # case_id
    "B": 70,   # customer_text
    "C": 70,   # historical reply
    "D": 16,   # prelabel
    "E": 18,   # prelabel confidence
    "F": 14,   # final intent
    "G": 18,   # confidence
    "H": 18,   # escalation
    "I": 35,   # escalation reason
    "J": 40,   # expected resolution
    "K": 45,   # review notes
    "L": 22,
    "M": 22,
    "N": 22,
    "O": 22,
}

for col, width in widths.items():
    ws.column_dimensions[col].width = width

# ---------------------------------------------------------
# Wrap text
# ---------------------------------------------------------

for row in ws.iter_rows():
    for cell in row:
        cell.alignment = Alignment(
            vertical="top",
            wrap_text=True
        )

# Make rows reasonably tall.
for row_num in range(2, len(df) + 2):
    ws.row_dimensions[row_num].height = 90

# ---------------------------------------------------------
# Add separate Guidelines sheet
# ---------------------------------------------------------

guide = wb.create_sheet("Annotation Guide")

guide_rows = [
    ["GOLDEN SET ANNOTATION GUIDE"],
    [""],
    ["Purpose"],
    [
        "Review the prelabel and decide the FINAL intent based on "
        "the customer's message and the historical Verizon response."
    ],
    [
        "The prelabel is only a suggestion. It is NOT ground truth."
    ],
    [""],
    ["Intent", "Definition"],
    ["I01", "Internet outage — internet/service unavailable or area outage."],
    ["I02", "Internet performance — slow, unstable, buffering, latency, poor speed."],
    ["I03", "Router/equipment — router, modem, gateway or service equipment problem."],
    ["I04", "TV/channel/service — TV, channels, programming or picture issues."],
    ["I05", "Mobile phone service — cellular/mobile phone, calls, data or wireless service."],
    ["I06", "FiOS app/access — FiOS app, login, authentication, credentials or access errors."],
    ["I07", "Billing/payment — bills, charges, payments, refunds, fees or pricing/account charges."],
    ["I08", "Installation/scheduling — installation, technician appointment or scheduling."],
    ["I09", "Voice/voicemail — home phone, landline, dial tone or voicemail."],
    ["I10", "Account/fraud — unauthorized activity, identity theft or genuine account security/fraud."],
    ["I11", "Service complaint — complaint primarily about Verizon/support/service experience."],
    ["I12", "Other/unclear — insufficient context, ambiguous, unrelated or no clear intent."],
    [""],
    ["Primary-intent rule"],
    [
        "If a message contains multiple issues, choose the issue that "
        "would determine the most important next support action."
    ],
    [
        "Example: 'Why am I still being billed and why is there an installation "
        "appointment?' → I07 if billing is the main unresolved issue."
    ],
    [""],
    ["Escalation"],
    [
        "Choose Yes when the case should be handed to a human because it "
        "requires account-specific action, sensitive handling, unresolved "
        "complex troubleshooting, fraud/security handling, or another action "
        "the AI should not perform autonomously."
    ],
    [
        "Choose No when the issue can reasonably be handled with a normal "
        "support response or troubleshooting step."
    ],
    [""],
    ["Confidence"],
    ["High", "Clear intent with little ambiguity."],
    ["Medium", "Likely intent but some ambiguity or multiple signals."],
    ["Low", "Difficult to determine from available context."],
]

for row in guide_rows:
    guide.append(row)

guide.column_dimensions["A"].width = 28
guide.column_dimensions["B"].width = 100

for row in guide.iter_rows():
    for cell in row:
        cell.alignment = Alignment(
            vertical="top",
            wrap_text=True
        )

for cell in guide[1]:
    cell.font = Font(
        bold=True,
        size=14
    )

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

wb.save(OUTPUT_FILE)

print("\n" + "=" * 60)
print("SUCCESS")
print("=" * 60)
print(f"Created: {OUTPUT_FILE}")
print(f"Rows: {len(df)}")
print("\nOpen golden_review.xlsx in Excel.")
print("Review FINAL intent, escalation, and resolution fields.")