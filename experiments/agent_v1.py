import re
import pickle
import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

RETRIEVER_PATH = "verizon_resolution_retriever_v3.pkl"

AUTO_HANDLE_THRESHOLD = 0.30
SAFE_GROUND_THRESHOLD = 0.20


# ============================================================
# INTENT TAXONOMY
# ============================================================

INTENTS = {
    "I01": "internet_outage",
    "I02": "internet_performance",
    "I03": "router_equipment",
    "I04": "tv_channel_service",
    "I05": "mobile_phone_service",
    "I06": "fios_app_access",
    "I07": "billing_payment",
    "I08": "installation_scheduling",
    "I09": "voice_voicemail",
    "I10": "account_fraud",
    "I11": "service_complaint",
    "I12": "other_unclear",
}


# ============================================================
# CLEANING
# ============================================================

def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


# ============================================================
# INTENT CLASSIFIER
# ============================================================

def assign_intent(text):

    t = clean_text(text)

    # --------------------------------------------------------
    # I10 — ACCOUNT FRAUD
    # --------------------------------------------------------

    fraud_words = [
        "fraud",
        "fraudulent",
        "unauthorized",
        "not authorize",
        "didn't authorize",
        "did not authorize",
        "stolen",
        "identity theft",
        "identity",
        "account takeover",
        "someone opened",
        "someone changed",
        "unknown charge",
        "unknown transaction",
        "unauthorized transaction",
        "unauthorized payment",
    ]

    if any(x in t for x in fraud_words):
        return "I10"


    # --------------------------------------------------------
    # I08 — INSTALLATION / TECHNICIAN
    # --------------------------------------------------------

    install_words = [
        "install",
        "installation",
        "installer",
        "technician",
        "technician appointment",
        "tech appointment",
        "appointment",
        "scheduled",
        "schedule a technician",
        "schedule technician",
        "no show",
        "no-show",
        "service visit",
        "technician visit",
    ]

    if any(x in t for x in install_words):
        return "I08"


    # --------------------------------------------------------
    # I09 — VOICE / VOICEMAIL
    # --------------------------------------------------------

    voice_words = [
        "voicemail",
        "voice mail",
        "landline",
        "home phone",
        "dial tone",
        "phone line",
        "telephone",
    ]

    if any(x in t for x in voice_words):
        return "I09"


    # --------------------------------------------------------
    # I07 — BILLING / PAYMENT
    # --------------------------------------------------------

    billing_words = [
        "bill",
        "billing",
        "charged",
        "charge",
        "payment",
        "paid",
        "paying",
        "refund",
        "overcharged",
        "price",
        "pricing",
        "fee",
        "fees",
        "tax",
        "credit card",
        "credit",
    ]

    if any(x in t for x in billing_words):
        return "I07"


    # --------------------------------------------------------
    # I06 — FIOS APP / ACCOUNT ACCESS
    # --------------------------------------------------------

    app_words = [
        "fios app",
        "my fios",
        "login",
        "log in",
        "sign in",
        "authentication",
        "password",
        "account access",
        "online account",
        "website login",
        "app login",
    ]

    if any(x in t for x in app_words):
        return "I06"


    # --------------------------------------------------------
    # I04 — TV / CHANNELS
    # --------------------------------------------------------

    tv_words = [
        "tv",
        "television",
        "channel",
        "channels",
        "dvr",
        "on demand",
        "ondemand",
        "hbo",
        "showtime",
        "univision",
        "sports channel",
        "cable",
        "programming",
        "picture quality",
        "pixelated",
    ]

    if any(x in t for x in tv_words):
        return "I04"


    # --------------------------------------------------------
    # I05 — MOBILE
    # --------------------------------------------------------

    mobile_words = [
        "mobile",
        "cell phone",
        "cellular",
        "wireless",
        "4g",
        "5g",
        "lte",
        "text messages",
        "text message",
        "sms",
        "data plan",
        "phone upgrade",
        "phone number",
    ]

    if any(x in t for x in mobile_words):
        return "I05"


    # --------------------------------------------------------
    # I01 — INTERNET OUTAGE
    # --------------------------------------------------------

    outage_words = [
        "no internet",
        "internet down",
        "internet is down",
        "internet not working",
        "internet stopped working",
        "can't connect",
        "cannot connect",
        "no connection",
        "service down",
        "outage",
        "all services down",
        "nothing loads",
        "internet outage",
    ]

    if any(x in t for x in outage_words):
        return "I01"


    # --------------------------------------------------------
    # I03 — ROUTER / EQUIPMENT
    # --------------------------------------------------------

    router_words = [
        "router",
        "modem",
        "ont",
        "battery backup",
        "bbu",
        "equipment",
        "red light",
        "reset router",
        "reboot router",
        "wired connection",
        "router keeps",
        "router is",
        "router keeps disconnecting",
    ]

    if any(x in t for x in router_words):
        return "I03"


    # --------------------------------------------------------
    # I02 — INTERNET PERFORMANCE
    # --------------------------------------------------------

    performance_words = [
        "slow",
        "slower",
        "slowest",
        "speed",
        "speeds",
        "slow internet",
        "slow speed",
        "slow speeds",
        "slow download",
        "slow upload",
        "buffering",
        "buffer",
        "lag",
        "latency",
        "packet loss",
        "disconnecting",
        "unstable",
        "poor speed",
        "speed test",
        "download speed",
        "upload speed",
        "dropping",
        "drops",
        "streaming",
        "streams",
    ]

    if any(x in t for x in performance_words):
        return "I02"


    # --------------------------------------------------------
    # I11 — SERVICE COMPLAINT
    # --------------------------------------------------------

    complaint_words = [
        "complaint",
        "ridiculous",
        "unacceptable",
        "terrible service",
        "worst service",
        "horrible service",
        "customer service",
        "customer support",
        "manager",
        "cancel",
        "cancellation",
        "canceling",
        "cancelled",
        "fed up",
        "unhappy",
        "disappointed",
    ]

    if any(x in t for x in complaint_words):
        return "I11"


    # --------------------------------------------------------
    # I12 — OTHER / UNCLEAR
    # --------------------------------------------------------

    return "I12"


# ============================================================
# LOAD RETRIEVER
# ============================================================

def load_retriever():

    with open(RETRIEVER_PATH, "rb") as f:
        return pickle.load(f)


# ============================================================
# RETRIEVE EVIDENCE
# ============================================================

def retrieve_evidence(customer_text, intent, retriever, top_k=3):

    # V3 stores intent-specific retrievers under
    # retriever["retrievers"].
    retrievers = retriever.get("retrievers", {})

    if intent not in retrievers:
        return []

    model = retrievers[intent]

    vectorizer = model["vectorizer"]
    matrix = model["matrix"]
    cases = model["cases"]

    query = clean_text(customer_text)

    if not query:
        return []

    query_vec = vectorizer.transform([query])

    scores = (matrix @ query_vec.T).toarray().ravel()

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for idx in top_indices:

        case = cases.iloc[idx]

        results.append({
            "historical_customer": str(case["customer_text"]),
            "historical_reply": str(case["support_reply"]),
            "similarity": float(scores[idx]),
            "response_type": str(
                case.get("response_type", "unknown")
            ),
        })

    return results


# ============================================================
# EVIDENCE QUALITY
# ============================================================

def evidence_is_safe(evidence):

    if not evidence:
        return False

    best = evidence[0]

    if best["similarity"] < SAFE_GROUND_THRESHOLD:
        return False

    bad_types = {
        "unknown",
        "generic",
        "escalation",
    }

    if best["response_type"] in bad_types:
        return False

    return True


# ============================================================
# ESCALATION
# ============================================================

def escalation_decision(intent, evidence):

    if intent == "I10":
        return (
            True,
            "Account fraud or unauthorized activity requires human review."
        )

    if intent == "I11":
        return (
            True,
            "Customer complaint requires human support intervention."
        )

    if intent == "I12":
        return (
            True,
            "Intent is unclear, so the agent should not guess."
        )

    if intent == "I08":
        return (
            True,
            "Installation or technician scheduling requires "
            "account-specific action."
        )

    if intent == "I07":
        return (
            True,
            "Billing or payment issues may require account-specific "
            "verification."
        )

    if not evidence:
        return (
            True,
            "No historical evidence was retrieved."
        )

    best = evidence[0]

    if best["similarity"] < SAFE_GROUND_THRESHOLD:
        return (
            True,
            f"Retrieved evidence was too weak to ground a reply "
            f"(similarity={best['similarity']:.3f})."
        )

    if not evidence_is_safe(evidence):
        return (
            True,
            "Retrieved evidence was not considered safe to use "
            "for an automated response."
        )

    if best["similarity"] < AUTO_HANDLE_THRESHOLD:
        return (
            True,
            f"Evidence quality was below the auto-handle threshold "
            f"(similarity={best['similarity']:.3f})."
        )

    return (
        False,
        f"Historical evidence is sufficiently similar and suitable "
        f"for grounding (similarity={best['similarity']:.3f})."
    )


# ============================================================
# REPLY
# ============================================================

def draft_reply(customer_text, intent, evidence, should_escalate):

    if should_escalate:

        if intent == "I10":
            return (
                "Thanks for reaching out. Because this involves account "
                "security or potentially unauthorized activity, we "
                "recommend having a support specialist review the account."
            )

        if intent == "I11":
            return (
                "I'm sorry you're dealing with this. A support specialist "
                "should review the issue and help with the next steps."
            )

        if intent == "I07":
            return (
                "Thanks for reaching out. Billing and payment issues can "
                "depend on the specific account, so a support specialist "
                "should review the details and help resolve this."
            )

        if intent == "I08":
            return (
                "Thanks for reaching out. Installation and technician "
                "appointments require account-specific information, so "
                "a support specialist should assist with this."
            )

        if intent == "I12":
            return (
                "I'd like to help, but I don't have enough information "
                "to determine the issue confidently. A support specialist "
                "can help clarify and resolve this."
            )

        return (
            "Thanks for reaching out. We don't have sufficiently reliable "
            "historical evidence to answer this confidently, so a support "
            "specialist should take a closer look."
        )

    best = evidence[0]

    historical_reply = best["historical_reply"]

    if not historical_reply or historical_reply.lower() == "nan":
        return (
            "Thanks for reaching out. We found a similar historical case, "
            "but its resolution is incomplete, so a support specialist "
            "should assist."
        )

    return historical_reply


# ============================================================
# FULL AGENT
# ============================================================

def run_agent(customer_text, retriever):

    intent = assign_intent(customer_text)

    evidence = retrieve_evidence(
        customer_text,
        intent,
        retriever,
        top_k=3,
    )

    should_escalate, reason = escalation_decision(
        intent,
        evidence,
    )

    reply = draft_reply(
        customer_text,
        intent,
        evidence,
        should_escalate,
    )

    return {
        "customer_text": customer_text,
        "intent": intent,
        "intent_name": INTENTS[intent],
        "should_escalate": should_escalate,
        "escalation_reason": reason,
        "reply": reply,
        "evidence": evidence,
    }


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("VERIZON SUPPORT AI AGENT V1")
    print("=" * 70)

    print("\nLoading retrieval model...")

    retriever = load_retriever()

    print("Agent ready.\n")

    demo_messages = [
        "My internet is extremely slow tonight",
        "Someone used my account and I did not authorize it",
        "My router keeps disconnecting",
        "I was charged twice on my bill",
        "My TV channel is pixelated",
        "I need to schedule a technician",
        "My voicemail isn't working",
        "This service is absolutely ridiculous",
    ]

    for message in demo_messages:

        result = run_agent(message, retriever)

        print("-" * 70)

        print("CUSTOMER:")
        print(message)

        print("\nINTENT:")
        print(
            result["intent"],
            "(" + result["intent_name"] + ")"
        )

        print("\nDECISION:")

        print(
            "ESCALATE"
            if result["should_escalate"]
            else "AUTO-HANDLE"
        )

        print("\nREASON:")
        print(result["escalation_reason"])

        print("\nREPLY:")
        print(result["reply"])

        print("\nTOP EVIDENCE:")

        if not result["evidence"]:
            print("No evidence retrieved.")

        for i, item in enumerate(result["evidence"], 1):

            print(
                f"{i}. similarity={item['similarity']:.3f} "
                f"type={item['response_type']}"
            )

            print(
                "   Historical customer:",
                item["historical_customer"][:180]
            )

            print(
                "   Historical reply:",
                item["historical_reply"][:180]
            )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)