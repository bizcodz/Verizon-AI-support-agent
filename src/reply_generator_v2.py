import re
import html
import joblib

from agent_v2_v24 import (
    assign_intent,
    retrieve_evidence,
    choose_grounding_evidence,
    escalation_decision,
)


RETRIEVER_FILE = "artifacts/verizon_resolution_retriever_v3.pkl"


# ------------------------------------------------------------
# Text cleanup
# ------------------------------------------------------------

def clean_text(text):
    if text is None:
        return ""

    text = html.unescape(str(text))
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def sanitize_historical_reply(reply):
    reply = clean_text(reply)

    # Never copy historical customer/agent handles.
    reply = re.sub(r"@\w+", "", reply)

    # Never copy historical agent signatures.
    reply = re.sub(r"\s+\^[A-Za-z]{2,5}\s*$", "", reply)

    return re.sub(r"\s+", " ", reply).strip()


# ------------------------------------------------------------
# Safe fallback replies
# ------------------------------------------------------------

def escalation_reply(intent):

    if intent == "I10":
        return (
            "Thanks for flagging this. Because this involves "
            "potential unauthorized activity, a human support "
            "agent should review the account securely."
        )

    if intent == "I11":
        return (
            "We're sorry for the frustration. This needs to "
            "be reviewed by a human support agent so they can "
            "look into the issue and assist you directly."
        )

    if intent == "I12":
        return (
            "Thanks for reaching out. We need a little more "
            "information to understand the issue before we "
            "can provide the right next step."
        )

    if intent == "I08":
        return (
            "Thanks for reaching out. Installation and "
            "technician scheduling require account-specific "
            "assistance, so a support agent should review "
            "the details with you."
        )

    if intent == "I07":
        return (
            "Thanks for reaching out. Billing and payment "
            "issues may require account-specific verification, "
            "so a support agent should review the account "
            "details with you."
        )

    return (
        "Thanks for reaching out. A support agent should "
        "review the details and help determine the appropriate "
        "next step."
    )


# ------------------------------------------------------------
# Generate grounded reply
# ------------------------------------------------------------

def generate_grounded_reply(
    customer_text,
    intent,
    evidence,
    should_escalate,
):

    # --------------------------------------------------------
    # Hard escalation policy
    # --------------------------------------------------------

    if should_escalate:
        return escalation_reply(intent)

    # --------------------------------------------------------
    # No evidence
    # --------------------------------------------------------

    if not evidence:
        return (
            "Thanks for reaching out. We don't have enough "
            "relevant historical information to safely suggest "
            "a resolution, so a support agent should review this."
        )

    # --------------------------------------------------------
    # Apply existing V2.4 grounding gates.
    # --------------------------------------------------------

    grounding = choose_grounding_evidence(
        evidence,
        customer_text,
        intent,
    )

    if grounding is None:
        return (
            "Thanks for reaching out. We don't have sufficiently "
            "relevant historical information to safely suggest "
            "a resolution, so a support agent should review this."
        )

    response_type = grounding["response_type"]

    historical_reply = sanitize_historical_reply(
        grounding["historical_reply"]
    )

    if not historical_reply:
        return (
            "Thanks for reaching out. We found related historical "
            "cases, but they did not contain a usable response. "
            "A support agent should review this."
        )

    # --------------------------------------------------------
    # Only use historical clarification when the system is
    # actually asking for clarification.
    # --------------------------------------------------------

    if response_type == "clarification":

        return historical_reply

    # --------------------------------------------------------
    # Historical troubleshooting / account-action response.
    #
    # These are the strongest forms of reusable resolution
    # evidence.
    # --------------------------------------------------------

    if response_type in {
        "troubleshooting",
        "account_action",
    }:

        return historical_reply

    # --------------------------------------------------------
    # Never blindly copy generic/unknown/escalation replies.
    # --------------------------------------------------------

    if response_type in {
        "unknown",
        "generic",
        "escalation",
    }:

        return (
            "Thanks for reaching out. We found a related "
            "historical case, but it does not provide enough "
            "actionable information to safely recommend a fix. "
            "A support agent should review this."
        )

    return (
        "Thanks for reaching out. A support agent should "
        "review the details and help determine the appropriate "
        "next step."
    )


# ------------------------------------------------------------
# Complete agent
# ------------------------------------------------------------

def run_agent(customer_text, retriever):

    intent = assign_intent(customer_text)

    evidence = retrieve_evidence(
        customer_text,
        intent,
        retriever,
        top_k=3,
    )

    selected_evidence = choose_grounding_evidence(
        evidence,
        customer_text,
        intent,
    )

    should_escalate, escalation_reason = escalation_decision(
        customer_text,
        intent,
        evidence,
    )

    reply = generate_grounded_reply(
        customer_text,
        intent,
        evidence,
        should_escalate,
    )

    return {
        "customer_text": customer_text,
        "predicted_intent": intent,
        "should_escalate": should_escalate,
        "escalation_reason": escalation_reason,
        "reply": reply,
        "selected_evidence": selected_evidence,
        "evidence_count": len(evidence),
    }


# ------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------

if __name__ == "__main__":

    print("Loading retriever...")
    retriever = joblib.load(RETRIEVER_FILE)

    test_messages = [
        "My internet is extremely slow and keeps buffering.",
        "My router has a red light.",
        "I was charged twice on my bill.",
        "Someone made an unauthorized charge on my account.",
        "My technician appointment was cancelled.",
        "Your customer service is terrible.",
    ]

    print()
    print("=" * 70)
    print("REPLY GENERATOR V2 SMOKE TEST")
    print("=" * 70)

    for message in test_messages:

        result = run_agent(
            message,
            retriever,
        )

        print()
        print("-" * 70)

        print("Customer:")
        print(message)

        print()
        print("Intent:")
        print(result["predicted_intent"])

        print()
        print("Escalate:")
        print(result["should_escalate"])

        print()
        print("Reason:")
        print(result["escalation_reason"])

        print()
        print("Reply:")
        print(result["reply"])

        print()
        print("Selected evidence:")

        evidence = result["selected_evidence"]

        if evidence is None:
            print("NONE")
        else:
            print(
                "Similarity:",
                f"{evidence['similarity']:.3f}"
            )

            print(
                "Response type:",
                evidence["response_type"]
            )

            print(
                "Historical customer:",
                evidence["historical_customer"]
            )

            print(
                "Historical reply:",
                evidence["historical_reply"]
            )
