\# VerizonSupport Intent Taxonomy



Version: 1.0



\## Purpose



Classify an incoming customer message according to the customer's

primary support need.



The label should represent the main issue the customer wants Verizon

to resolve, not every symptom mentioned in the message.



\---



\## I01 — internet\_outage



\### Definition

The customer reports that internet/service is unavailable or asks

whether there is an outage in their area.



\### Include

\- No internet

\- Internet completely unavailable

\- Area outage questions

\- Service outage reports

\- Connectivity suddenly stopped



\### Exclude

\- Slow internet with connectivity still working

\- Router replacement questions

\- TV-only problems



\### Examples

\- "No internet"

\- "Is there an outage in Queens?"

\- "My internet is down"



\---



\## I02 — internet\_performance



\### Definition

The customer has internet connectivity but reports poor performance,

slow speed, instability, or frequent drops.



\### Include

\- Slow internet

\- Speed below advertised level

\- Frequent disconnects

\- Poor Wi-Fi performance



\### Exclude

\- Complete outage

\- Pure router replacement request



\### Examples

\- "I pay for 150mbps but get 1.72mbps"

\- "Internet is extremely slow"



\---



\## I03 — router\_equipment



\### Definition

Problems or questions specifically concerning routers, modems,

or Verizon-provided networking equipment.



\### Include

\- Router failure

\- Router replacement

\- Router specifications

\- Equipment lights/errors

\- Modem/router troubleshooting



\### Exclude

\- General internet outage where equipment is only mentioned as a symptom



\### Examples

\- "My router stopped working"

\- "What router do you provide to new customers?"



\---



\## I04 — tv\_channel\_service



\### Definition

Problems or questions involving FiOS TV, channels, programming,

On Demand, blackouts, or TV access.



\### Include

\- Missing channels

\- TV blackouts

\- On Demand problems

\- TV programming questions

\- FiOS TV problems



\### Examples

\- "Why can't I watch this channel?"

\- "Is this channel blacked out?"



\---



\## I05 — mobile\_phone\_service



\### Definition

Problems with Verizon wireless/mobile phone service.



\### Include

\- Cannot make calls

\- Mobile data problems

\- Cellular signal problems

\- Phone service unavailable

\- Wireless service issues



\### Examples

\- "I can't make a phone call"

\- "My mobile data isn't working"



\---



\## I06 — fios\_app\_auth



\### Definition

Problems accessing FiOS apps, authentication, account verification,

or third-party services using Verizon/FiOS credentials.



\### Include

\- FiOS app problems

\- HBO Go authentication

\- Account verification for FiOS services

\- Login/access problems



\### Examples

\- "Can't access HBO Go using FiOS credentials"

\- "FiOS app is down"



\---



\## I07 — billing\_payment



\### Definition

Questions or complaints involving billing, charges, payments,

or amount due.



\### Include

\- Unexpected charges

\- Amount due

\- Payment problems

\- Incorrect bills

\- Billing disputes



\### Examples

\- "Why is my bill higher?"

\- "It says amount due $27.04"



\---



\## I08 — installation\_scheduling



\### Definition

Installation appointments, technician scheduling, arrival times,

or changing scheduled installation dates.



\### Include

\- Installation date

\- Technician appointment

\- Missed installation

\- Appointment changes



\### Examples

\- "Can I get an earlier installation?"

\- "Why hasn't the technician arrived?"



\---



\## I09 — voice\_voicemail



\### Definition

Verizon home phone or voicemail functionality problems.



\### Include

\- Voicemail unavailable

\- Voicemail activation

\- Home phone problems

\- Voice service issues



\### Examples

\- "Voicemail is not active"

\- "My home phone isn't working"



\---



\## I10 — account\_fraud



\### Definition

Unauthorized account creation, suspected fraud, identity theft,

or account activity the customer does not recognize.



\### Include

\- Fraudulent account

\- Unauthorized account

\- Identity-related Verizon account issue



\### Examples

\- "Someone opened an account in my name"

\- "I don't have a Verizon account but received a Verizon bill"



\---



\## I11 — service\_complaint



\### Definition

The customer's primary message is a complaint about Verizon's

customer service or support experience rather than a specific

technical/billing issue.



\### Include

\- Long hold complaints

\- Poor support experience

\- Repeated failed support attempts

\- Explicit customer service complaints



\### Exclude

If the customer clearly identifies a specific underlying problem,

label the underlying problem instead.



\### Examples

\- "Your customer service is terrible"

\- "I've been on hold for three hours"



\---



\## I12 — other\_unclear



\### Definition

The message does not contain enough information to confidently

assign one of the above intents.



\### Include

\- "Please help me"

\- Ambiguous messages

\- Unrelated messages

\- Messages requiring missing context



\### Rule



Do NOT force an uncertain message into another intent.



\---



\# General Labeling Rules



1\. Label the customer's primary support need.

2\. Do not use information that would not be available from the incoming

&#x20;  customer message itself.

3\. When multiple issues are present, choose the issue most directly

&#x20;  connected to the requested resolution.

4\. If no intent can be identified confidently, use `other\_unclear`.

5\. Fraud/security-related cases should be treated as escalation candidates.

6\. An intent label does not by itself determine whether a message is

&#x20;  safe to auto-handle.

