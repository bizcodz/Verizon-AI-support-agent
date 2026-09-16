\# VerizonSupport AI Support Agent



AI support agent built for the Hiver SDE Intern take-home assignment using the Kaggle Customer Support on Twitter dataset.



\## What it does



The agent:



1\. Classifies customer messages into 12 support intents.

2\. Retrieves relevant historical VerizonSupport interactions.

3\. Checks whether retrieved evidence is consistent and safe to use.

4\. Generates a grounded customer-facing response.

5\. Escalates sensitive, unclear, account-specific, or insufficiently supported cases.



\## Final Results



Evaluation on a 200-case golden set:



| Metric | Result |

|---|---:|

| Intent Accuracy | 70.0% |

| Macro F1 | 70.9% |

| Weighted F1 | 70.4% |

| Auto-handled | 33% |

| Escalated | 67% |

| Evidence available | 100% |

| Evidence selected | 59% |



Evidence benchmark:



| Metric | Top-1 | Top-3 |

|---|---:|---:|

| Relevant | 76% | 84% |

| Actionable | 66% | 78% |

| Safe to ground | 70% | 82% |



\## Setup



```bash

pip install -r requirements.txt

