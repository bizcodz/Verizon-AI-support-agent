# Decision Log

## 1. Brand selection — VerizonSupport

Selected VerizonSupport because it provides a substantial number of customer-support interactions with multi-turn troubleshooting and diverse service issues.

## 2. Intent discovery approach

Used exploratory TF-IDF + KMeans clustering to discover recurring themes before defining the final taxonomy. Clustering was treated as exploratory rather than as the final labeling mechanism because several clusters were noisy or conversational.

## 3. Frozen intent taxonomy

Defined 12 intents: internet outage, internet performance, router/equipment, TV/channel service, mobile phone service, FiOS app access, billing/payment, installation/scheduling, voice/voicemail, account/fraud, service complaint, and other/unclear.

## 4. Explicit I12 fallback

Added other_unclear so ambiguous, insufficient, or unrelated customer messages would not be forced into a specific operational intent.

## 5. Golden-set construction

Built a 200-case evaluation set from VerizonSupport customer-to-support interactions. Labels were human-approved/model-assisted rather than claimed as independently hand-labeled annotations.

## 6. First classifier baseline

Established a TF-IDF + Logistic Regression baseline to measure how far a simple supervised text classifier could go with limited labeled data.

## 7. Retrieval baseline

Added TF-IDF nearest-neighbor retrieval over historical Verizon customer-support cases. Historical cases were weakly labeled using rules, so retrieval results are evaluated with that limitation explicitly documented.

## 8. Separate evidence evaluation

Evaluated retrieval independently from classification using 50 golden cases and top-1/top-3 evidence judgments. This separates retrieval quality from downstream intent errors.

## 9. Top-3 evidence

Tested top-3 retrieval because lexical similarity alone can return superficially similar but operationally irrelevant cases. Top-3 improved relevant, actionable, and safe-to-ground evidence rates in the benchmark.

## 10. V2.3 escalation policy

Tested a highly conservative policy that only auto-handled troubleshooting/account-action evidence. It reduced automation coverage substantially without materially improving the measured safety proxy, so it was rejected.

## 11. V2.4 classifier change

Changed classification ordering to prioritize the customer's actual outcome before equipment mentions. In particular, outage/performance signals are evaluated before generic router mentions.

## 12. Conservative escalation

Billing, fraud/security, complaints, installation/account-specific issues, unclear cases, and insufficient evidence are routed to escalation rather than automatically handled.

## 13. No fabricated escalation ground truth

The 200-case set does not contain independently annotated human escalation labels. Therefore escalation agreement is reported only for the smaller judge-human calibration sample and is not presented as a 200-case gold metric.

## 14. Reply generation

Historical support replies are reused only when evidence passes similarity and consistency checks. Otherwise the system uses a conservative escalation fallback.

## 15. LLM judge calibration

Used an independent LLM judge on the successfully completed calibration cases and compared the same cases against human ratings using the same rubric. API free-tier quota limited the completed calibration sample to 20 cases.
