# Zoe — Private Insider Notes (Customer Compliance Confidences & Team Health)

## Private Customer Compliance Confidences (Shared Only with Zoe)
1. **St. Aurelia Health**:
   - **EU Residency & Isolation Clause**: Their legal team privately confirmed that **all patient data must stay in the EU** and run on **dedicated single-tenant hardware**. Any us-west-1, us-east-1, or shared multi-tenant hosting is an automatic contract breach.
2. **Klinik Nordhafen**:
   - **Sub-Processor Approval**: Every third-party service that touches patient data needs a **signed data processing agreement (DPA)** and **10 business days** of review before it goes live. No DPA means the service cannot be used.
3. **Dana (On-Call Lead)**:
   - **Burnout Recovery**: Privately told me she's recovering from burnout and **cannot take overnight on-call shifts this month**; the backup engineer, Ravi, can cover at most **one overnight shift per week**.

## Non-Negotiable Team Pact (Agreed in Last Retro)
- **No Planned Overnight Work**: No planned production work or scheduled checkpoints between **11:00 PM and 7:00 AM CET**. Any plan that needs an engineer awake overnight is vetoed unless Ravi takes it as his one shift.
- **Handoff Buffer**: Keep at least a **60-minute buffer** between finishing a migration and opening traffic to customers, so we can run smoke tests.
