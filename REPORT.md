\# VerizonSupport AI Support Agent — Take-Home Report



\## 1. Problem Framing



The goal is to build a support agent for VerizonSupport that can:



1\. classify an incoming customer message into a defined intent,

2\. retrieve historical support interactions that can provide evidence for a response,

3\. draft a response grounded in historical resolutions,

4\. decide whether the case can be auto-handled or should be escalated.



I selected VerizonSupport because the Customer Support on Twitter dataset contains substantial multi-turn Verizon interactions covering internet, TV, mobile, billing, equipment, installation, and account issues. The interactions provide useful historical troubleshooting and escalation evidence.



The final system uses a 12-intent taxonomy:



| ID | Intent |

|---|---|

| I01 | Internet outage |

| I02 | Internet performance |

| I03 | Router/equipment |

| I04 | TV/channel service |

| I05 | Mobile/phone service |

| I06 | FiOS app/access |

| I07 | Billing/payment |

| I08 | Installation/scheduling |

| I09 | Voice/voicemail |

| I10 | Account/fraud |

| I11 | Service complaint |

| I12 | Other/unclear |



I12 is intentionally conservative: messages without enough information to support a reliable intent are treated as unclear.



\---



\## 2. Data and Evaluation Setup



The source dataset contains approximately 2.81M tweets. VerizonSupport accounts for 17,966 tweets, including 17,851 parent-linked interactions and 10,585 responses.



A two-pass extraction process was used to construct customer → VerizonSupport cases. This produced 17,521 candidate cases.



The final golden set contains 200 cases sampled across the 12-intent taxonomy. The annotations were model-assisted and human-approved rather than independently hand-labeled from scratch.



The final 200-case set was kept separate from the larger historical retrieval corpus. Historical cases are used as the agent's knowledge base, while the golden cases are used for evaluation.



Three evaluation layers were used:



\- \*\*Intent classification:\*\* accuracy, macro F1, weighted F1.

\- \*\*Evidence retrieval:\*\* human judgments of relevance, actionability, and whether evidence was safe to ground a response on.

\- \*\*Reply quality:\*\* automated checks plus an LLM-judge calibration against human judgments.



The LLM judge calibration successfully evaluated 20 cases; the remaining planned cases could not be completed because of the Google API free-tier quota. No 200-case human escalation ground truth was fabricated.



\---



\## 3. Baselines and Final System



| System | Accuracy | Macro F1 | Weighted F1 | Notes |

|---|---:|---:|---:|---|

| TF-IDF + Logistic Regression | 47.5% | 45.5% | 46.9% | 160/40 split within golden set |

| TF-IDF historical retrieval | 68.5% | 68.8% | 68.5% | Uses weak historical labels |

| Final V2.4 agent | \*\*70.0%\*\* | \*\*70.9%\*\* | \*\*70.4%\*\* | Outcome-first classification + evidence gating |



The first classifier baseline demonstrates that a small labeled set is insufficient for robust classification. The retrieval baseline performs substantially better because historical Verizon interactions provide domain-specific lexical evidence.



The final V2.4 system adds domain-specific classification logic and evidence consistency checks on top of retrieval.



\### Final V2.4 results



\- Intent accuracy: \*\*70.0%\*\*

\- Macro F1: \*\*70.9%\*\*

\- Weighted F1: \*\*70.4%\*\*

\- Auto-handled: \*\*66/200 (33%)\*\*

\- Escalated: \*\*134/200 (67%)\*\*

\- Evidence available: \*\*200/200 (100%)\*\*

\- Evidence selected after safety/consistency checks: \*\*118/200 (59%)\*\*

\- Correct intent + auto-handle: \*\*50\*\*

\- Wrong intent + auto-handle: \*\*16\*\*

\- Auto-handle safety proxy: \*\*75.8%\*\*



The safety proxy is calculated from whether an automatically handled case has the correct intent. It is not equivalent to human-confirmed resolution safety.



\---



\## 4. System Architecture



The pipeline is:



```text

Customer message

&#x20;     |

&#x20;     v

Intent classification

&#x20;     |

&#x20;     v

Intent-conditioned retrieval

&#x20;     |

&#x20;     v

Top-3 historical cases

&#x20;     |

&#x20;     v

Evidence consistency + similarity gate

&#x20;     |

&#x20;     +------ insufficient / sensitive ------> Escalate

&#x20;     |

&#x20;     v

Sanitized historical response

&#x20;     |

&#x20;     v

Customer-facing draft

