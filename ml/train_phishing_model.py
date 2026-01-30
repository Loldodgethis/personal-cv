#!/usr/bin/env python3
"""
Train a small Logistic Regression model for phishing vs safe email classification.
Exports vocab + coefficients to JSON so the browser can run inference with no backend.

Option A: Install deps and run with sklearn (better accuracy):
  pip install numpy scikit-learn
  python ml/train_phishing_model.py

Option B: Run with stdlib only (no deps) — uses a simple hand-tuned export:
  python ml/train_phishing_model.py
"""
import json
import re
from pathlib import Path

# Inline minimal dataset: (text, label) where 1 = phishing, 0 = safe
SAMPLES = [
    ("Urgent: Verify your account now. Click here to avoid suspension.", 1),
    ("Your bank account has been locked. Enter your password to unlock.", 1),
    ("Congratulations! You have been selected for a free prize. Claim now.", 1),
    ("Click the link below to update your credentials immediately.", 1),
    ("Warning: Your account will expire in 24 hours. Sign in now to confirm.", 1),
    ("Verify your identity. We need your social security number to proceed.", 1),
    ("Act now! Your credit card may be compromised. Log in to secure your account.", 1),
    ("You have won a million dollars. Click here to claim your reward.", 1),
    ("Suspicious activity detected. Confirm your login and password now.", 1),
    ("Update your payment info. Enter your CVV and card number at this link.", 1),
    ("ASAP: Your account will be suspended. Verify now by clicking the link.", 1),
    ("Dear winner, you have been selected. Claim your free gift before it expires.", 1),
    ("We need to verify your bank details. Please confirm your account.", 1),
    ("Immediate action required. Sign in to avoid permanent account closure.", 1),
    ("Click here to reset your password. Ignore this and your account may be locked.", 1),
    ("Hi, just checking in about our meeting tomorrow at 3pm.", 0),
    ("Thanks for sending the report. I'll review it and get back to you.", 0),
    ("The project deadline is next Friday. Let me know if you need an extension.", 0),
    ("Please find attached the invoice for your records.", 0),
    ("We're having a team lunch on Thursday. Can you make it?", 0),
    ("I wanted to follow up on the proposal we discussed last week.", 0),
    ("Could you send me the slides from the presentation when you get a chance?", 0),
    ("The conference has been moved to March. I'll send the updated invite.", 0),
    ("Reminder: the all-hands is at 10am. See you there.", 0),
    ("Here are the meeting notes. Let me know if anything is missing.", 0),
    ("Thanks for your help on the budget. I'll circulate the final version.", 0),
    ("The new hire starts Monday. I'll introduce you at standup.", 0),
    ("Can we reschedule our 1:1 to Wednesday? Something came up.", 0),
    ("I've shared the folder with you. You should have edit access now.", 0),
    ("Quick question about the API — do we support pagination?", 0),
    # Invoice / attachment malware phishing
    (
        "Please see the attached invoice for your recent transaction. Important: Payment is OVERDUE "
        "and must be reviewed today to avoid penalties. Attachment: Invoice_013889.exe. "
        "Open the attachment and follow the instructions to confirm payment. Accounting Team.",
        1,
    ),
    # FP-1..FP-6: Legitimate high-risk-style (safe — real security alerts, university IT, payroll, DocuSign, Drive, bank)
    (
        "We blocked a sign-in attempt to your account because it didn't match your usual activity. "
        "If this was you, no action is needed. If this wasn't you, please secure your account by visiting: "
        "https://account.microsoft.com/security Thanks, Microsoft Account Team.",
        0,
    ),
    (
        "Your university password will expire in 2 days. To avoid interruption, please update your password before expiration. "
        "Reset here: https://passwords.university.edu/reset If you have already changed your password, you may ignore this message. IT Help Desk.",
        0,
    ),
    (
        "As part of our annual payroll audit, employees are required to review their direct deposit information. "
        "Please log in to the employee portal by Friday to confirm or update your details: https://workday.companyname.com "
        "No changes are required if your information is already correct. Payroll Operations.",
        0,
    ),
    (
        "You have been requested to review and sign the following document: Internship Agreement. "
        "Review & Sign: https://docusign.net/Signing/start.aspx Do not share this email. DocuSign.",
        0,
    ),
    (
        "Vishal shared a file with you: CS_Project_Final.pdf. Open: https://drive.google.com/file/d/1F9x... "
        "You can access this file until Feb 5. Google Drive.",
        0,
    ),
    (
        "We declined a transaction of $842.19 at BestBuy.com. If this was you, you can retry the purchase. "
        "If this wasn't you, please review activity here: https://www.chase.com/security Chase Fraud Prevention.",
        0,
    ),
    # NM-1..NM-7: Legitimate near-miss (single mutation → phish: 1-word threat flip, deadline edge, CTA edge, etc.)
    (
        "We detected a new sign-in to your account from a device we haven't seen before. "
        "If this was you, no action is required. If this wasn't you, please review your recent activity: "
        "https://account.google.com/security Thanks, Google Account Security.",
        0,
    ),
    (
        "Our systems recommend updating your password periodically to maintain account security. "
        "You may update your password here at your convenience: https://login.microsoftonline.com — IT Services.",
        0,
    ),
    (
        "Vishal shared a document with you: Research_Proposal_Draft.pdf. View file: https://drive.google.com/file/d/abc123 Google Drive.",
        0,
    ),
    (
        "Your direct deposit information is available for review in the employee portal. Access here: https://workday.company.com "
        "No changes are required unless your information is outdated. Payroll Team.",
        0,
    ),
    (
        "We declined a transaction of $412.77 at Amazon. If this was you, you can retry the purchase. "
        "If this wasn't you, please visit your bank app or website to review activity. Chase Fraud Prevention.",
        0,
    ),
    (
        "You have been requested to review and sign a document. Review: https://docusign.net/Signing/start.aspx DocuSign.",
        0,
    ),
    (
        "IT will be performing scheduled maintenance this weekend. Details are available on the IT portal: "
        "https://it.university.edu/status Thank you, IT Services.",
        0,
    ),
    # SAFE-1..SAFE-7: Informational anchor (no urgency, no threat, no "click now", optional navigation only)
    (
        "A new device signed in to your account on Jan 29 at 6:18 PM. Device: Chrome on macOS. Location: California, United States. "
        "If this was you, no further action is required. You can review account activity anytime by visiting your account settings. Account Security Team.",
        0,
    ),
    (
        "This message is to inform you of upcoming system updates scheduled for the Spring semester. "
        "Details are available on the university IT website: https://it.university.edu/updates No action is required. University IT Services.",
        0,
    ),
    (
        "Your payroll for the pay period ending Jan 26 has been processed successfully. "
        "Funds will be deposited according to your existing direct deposit instructions. No further action is required. Payroll Team.",
        0,
    ),
    (
        "Vishal shared a file with you: CS440_Project_Report.pdf. You can access this file using your existing Google Drive account. Google Drive.",
        0,
    ),
    (
        "This email confirms that the following document has been completed: Document: Internship Agreement. Status: Completed. "
        "No further action is required. DocuSign.",
        0,
    ),
    (
        "Your January bank statement is now available. To view your statement, please sign in through your bank's official website or mobile app. "
        "This message is for informational purposes only. Your Bank.",
        0,
    ),
    (
        "This week's engineering updates are now posted on Confluence. There are no action items required for this update. Thanks, Engineering Manager.",
        0,
    ),
]


def tokenize(text):
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [w for w in text.split() if w]


def export_with_sklearn():
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import CountVectorizer

    texts = [t for t, _ in SAMPLES]
    labels = [l for _, l in SAMPLES]
    vectorizer = CountVectorizer(lowercase=True, token_pattern=r"\b\w+\b", max_features=500)
    X = vectorizer.fit_transform(texts)
    y = np.array(labels, dtype=np.float64)
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X, y)
    vocab = vectorizer.get_feature_names_out().tolist()
    coef = model.coef_[0].tolist()
    intercept = float(model.intercept_[0])
    return {"vocab": vocab, "coef": coef, "intercept": intercept}


def export_hand_tuned():
    """No sklearn: export hand-tuned linear model using taxonomy (urgency, threats, authority, credentials, CTA, financial, attachment, etc.)."""
    # 1–2: Urgency & threats
    v1 = ["urgent", "immediately", "action", "required", "final", "notice", "warning", "deadline", "overdue", "penalties", "suspend", "suspended", "expire", "compromised", "closure", "permanent", "temporarily", "violation", "unauthorized", "suspicious", "activity", "breach", "restricted", "frozen", "funds", "legal"]
    c1 = [1.6, 1.5, 1.2, 1.0, 1.3, 1.2, 1.1, 1.0, 1.4, 1.3, 1.4, 1.5, 1.1, 1.3, 1.2, 1.0, 1.2, 1.1, 1.2, 1.3, 1.0, 1.2, 1.1, 1.0, 1.2, 1.0]
    # 3–4: Authority & credentials
    v2 = ["security", "team", "department", "support", "billing", "accounting", "administrator", "compliance", "payroll", "secure", "server", "verify", "confirm", "identity", "password", "credentials", "login", "authenticate", "validate", "reset", "authentication"]
    c2 = [1.0, 0.9, 0.8, 0.8, 1.1, 1.0, 1.0, 1.0, 1.0, 1.1, 1.0, 1.5, 1.4, 1.3, 1.6, 1.4, 1.2, 1.1, 1.1, 1.1, 1.0]
    # 5–6: CTA & financial
    v3 = ["click", "link", "open", "attachment", "download", "review", "restore", "access", "payment", "invoice", "attached", "transaction", "refund", "charge", "wire", "transfer", "billing", "overdue", "penalties"]
    c3 = [1.4, 0.9, 1.3, 1.5, 1.2, 1.0, 1.2, 1.0, 1.3, 1.0, 1.4, 0.9, 1.0, 1.0, 1.1, 0.9, 1.0, 1.4, 1.3]
    # 7–8: Attachment & file
    v4 = ["attachment", "attached", "document", "file", "open", "instructions", "exe", "secure", "encrypted", "macros", "editing", "compressed"]
    c4 = [1.5, 1.4, 1.0, 1.0, 1.2, 1.1, 1.8, 1.0, 1.0, 1.3, 1.2, 1.0]
    # Existing + safe (negative)
    v5 = ["account", "bank", "credit", "claim", "reward", "free", "winner", "prize", "congratulations", "gift", "dear", "thanks", "meeting", "report", "project", "invoice", "follow", "proposal", "presentation", "conference", "reminder", "notes", "budget", "folder", "question", "extension", "shared", "reschedule", "update", "asap", "locked", "social", "cvv", "selected", "ownership", "protection"]
    c5 = [0.6, 1.2, 1.1, 1.2, 1.1, 1.0, 1.1, 1.0, 1.0, 1.0, 0.3, -0.5, -0.6, -0.5, -0.5, -0.4, -0.5, -0.5, -0.5, -0.5, -0.4, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 1.0, 1.2, 1.0, 1.0, 0.9, 1.0, 1.0, 1.0]

    # Merge: for duplicate words keep max coefficient (strongest signal)
    merged = {}
    for vlist, clist in [(v1, c1), (v2, c2), (v3, c3), (v4, c4), (v5, c5)]:
        for w, c in zip(vlist, clist):
            merged[w] = max(merged.get(w, c), c)
    vocab = list(merged.keys())
    coef = [merged[w] for w in vocab]
    intercept = -2.5  # bias toward Safe so we don't overfit to keywords (SAFE-1..SAFE-7 anchor)
    return {"vocab": vocab, "coef": coef, "intercept": intercept}


def main():
    try:
        out = export_with_sklearn()
        print("Exported model using sklearn.")
    except ImportError:
        out = export_hand_tuned()
        print("Exported hand-tuned model (install numpy + scikit-learn for trained model).")
    out_path = Path(__file__).resolve().parent.parent / "js" / "safemail-model.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print("Wrote", out_path, "| vocab size:", len(out["vocab"]))


if __name__ == "__main__":
    main()
