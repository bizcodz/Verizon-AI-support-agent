import re
import pickle
import numpy as np


# ============================================================
# CONFIG
# ============================================================

RETRIEVER_PATH = "verizon_resolution_retriever_v3.pkl"

AUTO_HANDLE_THRESHOLD = 0.30
SAFE_GROUND_THRESHOLD = 0.20


# ============================================================
# TAXONOMY
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
# TEXT CLEANING
# ============================================================

def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


# ============================================================
# HELPER
# ============================================================

def contains_any(text, phrases):
    return any(p in text for p in phrases)


# ============================================================
# V2 INTENT CLASSIFIER
#
# Main improvement over V1:
# - explicit priority ordering
# - compound/context rules
# - avoids treating "mobile" inside "FiOS mobile app"
#   as mobile-service intent
# - distinguishes internet outage from equipment symptoms
# ============================================================

def assign_intent(text):

    t = clean_text(text)


    # --------------------------------------------------------
    # Very short / conversational messages
    # --------------------------------------------------------

    tokens = t.split()

    if len(tokens) <= 2:
        vague = {
            "yes",
            "no",
            "same",
            "thanks",
            "thank you",
            "dm",
            "dm?",
            "help",
            "hello",
            "hi",
        }

        if t in vague:
            return "I12"


    # --------------------------------------------------------
    # I10 — ACCOUNT FRAUD / SECURITY
    # Highest priority because security-sensitive language
    # should not be overridden by billing/mobile keywords.
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
        "scammed",
        "scam",
        "suspicious",
    ]

    if contains_any(t, fraud_words):
        return "I10"


    # --------------------------------------------------------
    # I06 — FIOS APP / ONLINE ACCESS
    #
    # Must come before I05 because:
    # "FiOS mobile app" is an app-access problem, not
    # mobile-service support.
    # --------------------------------------------------------

    app_phrases = [
        "fios app",
        "my fios",
        "fios mobile app",
        "app is",
        "app isn't",
        "app isnt",
        "app not",
        "app down",
        "app error",
        "app issue",
        "app problem",
        "login",
        "log in",
        "sign in",
        "authentication",
        "password",
        "account access",
        "online account",
        "website login",
        "accessing hbo go",
        "hbo go",
    ]

    if contains_any(t, app_phrases):
        return "I06"


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

    if contains_any(t, voice_words):
        return "I09"


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
        "service appointment",
        "service visit",
        "technician visit",
        "no show",
        "no-show",
    ]

    if contains_any(t, install_words):
        return "I08"


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
        "pricing",
        "price",
        "fee",
        "fees",
        "tax",
        "credit card",
        "credit limit",
        "payment plan",
        "final bill",
    ]

    if contains_any(t, billing_words):
        return "I07"


    # --------------------------------------------------------
    # I04 — TV / CHANNEL / DVR
    #
    # Includes explicit TV vocabulary and common TV-service
    # symptoms such as pixelation.
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
        "pixelation",
        "tv freezes",
        "tv freeze",
        "recording",
    ]

    if contains_any(t, tv_words):
        return "I04"


    # --------------------------------------------------------
    # I01 — INTERNET OUTAGE / CONNECTIVITY
    #
    # Connectivity failure gets priority over generic router
    # vocabulary.
    # --------------------------------------------------------

    outage_phrases = [
        "no internet",
        "internet down",
        "internet is down",
        "internet not working",
        "internet stopped working",
        "internet out",
        "internet outage",
        "no connection",
        "can't connect",
        "cannot connect",
        "can't get online",
        "cannot get online",
        "nothing loads",
        "service down",
        "service outage",
        "all services down",
        "outage",
        "offline",
        "no service",
    ]

    if contains_any(t, outage_phrases):

        # If the message explicitly describes a complete
        # connectivity failure, treat it as outage even if
        # router/equipment is mentioned.
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
        "router light",
        "reset router",
        "reboot router",
        "factory reset",
        "wired connection",
        "router keeps",
        "router is",
        "router issue",
        "router problem",
    ]

    if contains_any(t, router_words):
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
        "unstable",
        "poor speed",
        "speed test",
        "download speed",
        "upload speed",
        "dropping",
        "drops",
        "streaming",
        "streams",
        "disconnecting",
        "performance",
    ]

    if contains_any(t, performance_words):
        return "I02"


    # --------------------------------------------------------
    # I05 — MOBILE / WIRELESS
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
        "wireless service",
        "mobile service",
    ]

    if contains_any(t, mobile_words):
        return "I05"


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
        "done with",
        "worst",
    ]

    if contains_any(t, complaint_words):
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
            "historical_customer": str(
                case["customer_text"]
            ),
            "historical_reply": str(
                case["support_reply"]
            ),
            "similarity": float(
                scores[idx]
            ),
            "response_type": str(
                case.get(
                    "response_type",
                    "unknown"
                )
            ),
        })

    return results


# ============================================================
# EVIDENCE RANKING
#
# Instead of trusting similarity alone, prefer response
# types that contain a potential action or troubleshooting
# step.
# ============================================================

RESPONSE_PRIORITY = {
    "troubleshooting": 4,
    "account_action": 4,
    "clarification": 2,
    "escalation": 1,
    "generic": 0,
    "unknown": 0,
}


def rank_evidence(evidence):

    if not evidence:
        return []

    # Retrieval order is primarily determined by similarity.
    # Response type is evaluated later by the safety gate.
    return sorted(
        evidence,
        key=lambda x: x["similarity"],
        reverse=True,
    )



def normalize_text(text):
    """Normalize text for lightweight evidence consistency checks."""
    text = str(text).lower()

    replacements = {
        "wi-fi": "wifi",
        "fios": "fios",
        "fi os": "fios",
        "internet connection": "internet",
        "service outage": "outage",
        "not working": "down",
        "isn't working": "down",
        "is not working": "down",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def get_problem_signals(text):
    """
    Extract fine-grained problem signals from a support message.

    These signals are used only as an evidence-consistency gate.
    They are intentionally interpretable rather than being the
    primary intent classifier.
    """

    text = normalize_text(text)

    signals = set()

    signal_groups = {

        # ----------------------------------------------------
        # INTERNET
        # ----------------------------------------------------

        "internet_outage": [
            "internet down",
            "internet is down",
            "no internet",
            "no connection",
            "nothing loads",
            "internet out",
            "internet outage",
            "service down",
            "service outage",
            "offline",
            "no service",
            "completely down",
        ],

        "internet_slow": [
            "slow",
            "sluggish",
            "poor speed",
            "low speed",
            "slow internet",
            "slow speed",
            "slow speeds",
            "slow download",
            "slow upload",
            "buffering",
            "buffer",
            "latency",
            "packet loss",
            "poor performance",
            "speed test",
            "download speed",
            "upload speed",
        ],

        "internet_disconnect": [
            "disconnect",
            "disconnecting",
            "keeps going out",
            "keeps dropping",
            "dropping",
            "drops",
            "unstable",
        ],

        # ----------------------------------------------------
        # ROUTER / EQUIPMENT
        # ----------------------------------------------------

        "router": [
            "router",
            "modem",
            "ont",
            "gateway",
            "router light",
            "red light",
            "battery backup",
            "bbu",
        ],

        # ----------------------------------------------------
        # TV
        # ----------------------------------------------------

        "tv_general": [
            "tv",
            "television",
            "channel",
            "channels",
            "cable",
            "programming",
            "picture",
            "picture quality",
            "pixelated",
            "pixelation",
            "tv freezes",
            "tv freeze",
            "recording",
            "dvr",
            "on demand",
            "ondemand",
            "vod",
        ],

        "tv_content": [
            "show",
            "movie",
            "program",
            "episode",
            "star trek",
            "discovery",
            "content",
            "title",
        ],

        # ----------------------------------------------------
        # APP
        # ----------------------------------------------------

        "app_outage": [
            "app is down",
            "app down",
            "app isn't working",
            "app isnt working",
            "app not working",
            "app is unavailable",
            "app unavailable",
            "app outage",
            "app stopped working",
        ],

        "app_login": [
            "login",
            "log in",
            "sign in",
            "authentication",
            "password",
            "account access",
            "website login",
        ],

        "app_general": [
            "app",
            "application",
            "fios app",
            "fios mobile app",
            "my fios",
        ],

        # ----------------------------------------------------
        # MOBILE
        # ----------------------------------------------------

        "mobile_service": [
            "mobile service",
            "cellular",
            "cell phone",
            "phone upgrade",
            "data plan",
            "wireless service",
            "4g",
            "5g",
            "lte",
            "text message",
            "text messages",
            "sms",
        ],

        # ----------------------------------------------------
        # BILLING
        # ----------------------------------------------------

        "billing": [
            "bill",
            "billing",
            "charged",
            "charge",
            "payment",
            "paid",
            "paying",
            "refund",
            "overcharged",
            "pricing",
            "price",
            "fee",
            "fees",
            "tax",
            "credit card",
            "credit limit",
            "payment plan",
            "final bill",
        ],

        # ----------------------------------------------------
        # TECHNICIAN
        # ----------------------------------------------------

        "technician": [
            "technician",
            "tech",
            "service visit",
            "appointment",
            "schedule",
            "scheduled",
            "installation",
            "install",
        ],

        # ----------------------------------------------------
        # VOICEMAIL
        # ----------------------------------------------------

        "voicemail": [
            "voicemail",
            "voice mail",
            "voicemail transcript",
        ],

        # ----------------------------------------------------
        # LANDLINE
        # ----------------------------------------------------

        "landline": [
            "landline",
            "home phone",
            "dial tone",
            "phone line",
            "telephone",
        ],

        # ----------------------------------------------------
        # FRAUD
        # ----------------------------------------------------

        "fraud": [
            "fraud",
            "fraudulent",
            "unauthorized",
            "did not authorize",
            "didn't authorize",
            "not authorized",
            "identity theft",
            "account takeover",
            "someone used my account",
            "unknown transaction",
            "unauthorized transaction",
            "unauthorized payment",
            "scam",
            "scammed",
            "suspicious",
        ],

        # ----------------------------------------------------
        # COMPLAINT
        # ----------------------------------------------------

        "complaint": [
            "ridiculous",
            "horrible",
            "terrible",
            "thieves",
            "worst",
            "awful",
            "disappointed",
            "bad customer service",
            "customer service",
            "customer support",
            "unacceptable",
        ],
    }

    for signal, phrases in signal_groups.items():
        if any(phrase in text for phrase in phrases):
            signals.add(signal)

    return signals


def evidence_is_consistent(
    customer_text,
    historical_customer_text,
    intent,
):
    """
    Conservative evidence-consistency gate.

    Retrieval similarity is not enough. The historical customer
    problem must also match the current problem subtype.

    This prevents cases such as:
        "Fios app is down"
    from grounding on:
        "Star Trek Discovery is missing from Fios app"
    """

    current = get_problem_signals(customer_text)
    historical = get_problem_signals(historical_customer_text)

    if not current or not historical:
        return False

    # --------------------------------------------------------
    # INTENT-SPECIFIC COMPATIBILITY
    # --------------------------------------------------------

    if intent == "I01":
        # Current outage should match another outage.
        return (
            "internet_outage" in current
            and "internet_outage" in historical
        )

    if intent == "I02":
        # Slow internet should match slow/performance cases.
        current_performance = {
            "internet_slow",
            "internet_disconnect",
        }

        historical_performance = {
            "internet_slow",
            "internet_disconnect",
        }

        return bool(
            current & current_performance
        ) and bool(
            historical & historical_performance
        )

    if intent == "I03":
        # Router/equipment cases require router/equipment evidence.
        return (
            "router" in current
            and "router" in historical
        )

    if intent == "I04":
        # TV cases require TV evidence.
        if "tv_general" not in current:
            return False

        if "tv_general" not in historical:
            return False

        # If current is specifically about content availability,
        # prefer another content-related case.
        if "tv_content" in current:
            return "tv_content" in historical

        return True

    if intent == "I05":
        return (
            "mobile_service" in current
            and "mobile_service" in historical
        )

    if intent == "I06":

        # App outage must match another app-outage case.
        if "app_outage" in current:
            return "app_outage" in historical

        # Login/access problems should match access cases.
        if "app_login" in current:
            return "app_login" in historical

        # Generic app problem can match app evidence,
        # but not a completely different TV-content issue.
        return (
            "app_general" in current
            and (
                "app_general" in historical
                or "app_login" in historical
                or "app_outage" in historical
            )
        )

    if intent == "I07":
        return (
            "billing" in current
            and "billing" in historical
        )

    if intent == "I08":
        return (
            "technician" in current
            and "technician" in historical
        )

    if intent == "I09":
        # A voicemail problem should not be grounded on a generic
        # landline outage.
        return (
            "voicemail" in current
            and "voicemail" in historical
        )

    if intent == "I10":
        return (
            "fraud" in current
            and "fraud" in historical
        )

    if intent == "I11":
        return (
            "complaint" in current
            and "complaint" in historical
        )

    # I12 should not be autonomously grounded.
    return False


# ============================================================
# FIND USABLE EVIDENCE
# ============================================================

def choose_grounding_evidence(
    evidence,
    customer_text,
    intent,
):

    if not evidence:
        return None

    candidates = []

    for item in evidence:

        if not isinstance(item, dict):
            raise TypeError(
                "choose_grounding_evidence expected "
                "evidence dictionaries, but received: "
                f"{type(item).__name__}: {item!r}"
            )

        similarity = item["similarity"]
        response_type = item["response_type"]
        historical_customer = item["historical_customer"]

        # ----------------------------------------------------
        # Gate 1: minimum retrieval similarity
        # ----------------------------------------------------

        if similarity < SAFE_GROUND_THRESHOLD:
            continue

        # ----------------------------------------------------
        # Gate 2: problem consistency
        # ----------------------------------------------------

        if not evidence_is_consistent(
            customer_text,
            historical_customer,
            intent,
        ):
            continue

        # ----------------------------------------------------
        # Small response-quality bonus
        # ----------------------------------------------------

        if response_type in {
            "troubleshooting",
            "account_action",
        }:
            quality_bonus = 0.08

        elif response_type == "clarification":
            quality_bonus = 0.04

        elif response_type == "escalation":
            quality_bonus = 0.02

        else:
            quality_bonus = 0.0

        grounding_score = (
            similarity + quality_bonus
        )

        candidates.append({
            **item,
            "grounding_score": grounding_score,
        })

    # No evidence survived both gates.
    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item["grounding_score"],
        reverse=True,
    )

    best = candidates[0]

    # --------------------------------------------------------
    # Final confidence gate
    # --------------------------------------------------------

    if best["similarity"] < AUTO_HANDLE_THRESHOLD:
        return None

    # Generic/unknown replies need stronger evidence.
    if best["response_type"] in {
        "unknown",
        "generic",
    }:
        if best["similarity"] < 0.60:
            return None

    return best


# ============================================================
# ESCALATION POLICY
# ============================================================

def escalation_decision(customer_text,intent, evidence):

    # Security-sensitive
    if intent == "I10":
        return (
            True,
            "Account fraud or unauthorized activity "
            "requires human review."
        )

    # Complaints
    if intent == "I11":
        return (
            True,
            "Customer complaint requires human "
            "support intervention."
        )

    # Ambiguous
    if intent == "I12":
        return (
            True,
            "Intent is unclear, so the agent should "
            "not guess."
        )

    # Account-specific operations
    if intent == "I08":
        return (
            True,
            "Installation or technician scheduling "
            "requires account-specific action."
        )

    if intent == "I07":
        return (
            True,
            "Billing or payment issues may require "
            "account-specific verification."
        )

    if not evidence:
        return (
            True,
            "No historical evidence was retrieved."
        )

    grounding = choose_grounding_evidence(
        evidence,
        customer_text,
        intent,
    )

    if grounding is None:
        best_similarity = max(
            x["similarity"]
            for x in evidence
        )

        return (
            True,
            "Retrieved evidence did not contain a "
            "sufficiently actionable and safe historical "
            f"resolution (best similarity="
            f"{best_similarity:.3f})."
        )

    if grounding["similarity"] < AUTO_HANDLE_THRESHOLD:
        return (
            True,
            "Historical evidence was below the "
            "auto-handle confidence threshold."
        )

    return (
        False,
        "Historical troubleshooting/action evidence "
        "was sufficiently similar and suitable for "
        "grounding."
    )


# ============================================================
# REPLY GENERATION
# ============================================================

def draft_reply(
    customer_text,
    intent,
    evidence,
    should_escalate
):

    if should_escalate:

        if intent == "I10":
            return (
                "Thanks for reaching out. Because this "
                "involves account security or potentially "
                "unauthorized activity, we recommend having "
                "a support specialist review the account."
            )

        if intent == "I11":
            return (
                "I'm sorry you're dealing with this. "
                "A support specialist should review the "
                "issue and help with the next steps."
            )

        if intent == "I07":
            return (
                "Thanks for reaching out. Billing and "
                "payment issues can depend on the specific "
                "account, so a support specialist should "
                "review the details and help resolve this."
            )

        if intent == "I08":
            return (
                "Thanks for reaching out. Installation and "
                "technician appointments require "
                "account-specific information, so a support "
                "specialist should assist with this."
            )

        if intent == "I12":
            return (
                "I'd like to help, but I don't have enough "
                "information to determine the issue "
                "confidently. A support specialist can help "
                "clarify and resolve this."
            )

        return (
            "Thanks for reaching out. We don't have "
            "sufficiently reliable historical evidence to "
            "answer this confidently, so a support "
            "specialist should take a closer look."
        )


    grounding = choose_grounding_evidence(
        evidence,
        customer_text,
        intent,
    )

    if grounding is None:
        return (
            "Thanks for reaching out. We don't have "
            "sufficiently reliable historical evidence to "
            "answer this confidently, so a support "
            "specialist should take a closer look."
        )

    return grounding["historical_reply"]


# ============================================================
# FULL AGENT
# ============================================================

def run_agent(customer_text, retriever):

    intent = assign_intent(
        customer_text
    )

    evidence = retrieve_evidence(
        customer_text,
        intent,
        retriever,
        top_k=3,
    )

    evidence = rank_evidence(
        evidence
    )

    should_escalate, reason = escalation_decision(
        customer_text,
        intent,
        evidence,
    )

    reply = draft_reply(
        customer_text,
        intent,
        evidence,
        should_escalate,
    )

    grounding = choose_grounding_evidence(
        evidence,
        customer_text,
        intent,
    )

    return {
        "customer_text": customer_text,
        "intent": intent,
        "intent_name": INTENTS[intent],
        "should_escalate": should_escalate,
        "escalation_reason": reason,
        "reply": reply,
        "evidence": evidence,
        "grounding_evidence": grounding,
    }


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("VERIZON SUPPORT AI AGENT V2")
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
        "The Fios mobile app is down",
        "My internet is completely down and the router shows no connection",
    ]

    for message in demo_messages:

        result = run_agent(
            message,
            retriever
        )

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
        print(
            result["escalation_reason"]
        )

        print("\nREPLY:")
        print(result["reply"])

        print("\nTOP EVIDENCE:")

        if not result["evidence"]:
            print("No evidence retrieved.")

        for i, item in enumerate(
            result["evidence"],
            1
        ):

            print(
                f"{i}. similarity="
                f"{item['similarity']:.3f} "
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

        if result["grounding_evidence"]:

            print("\nSELECTED GROUNDING EVIDENCE:")

            g = result["grounding_evidence"]

            print(
                f"similarity={g['similarity']:.3f} "
                f"type={g['response_type']}"
            )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)