# Maya — Private Insider Notes (Launch Budget, Vendor Quotes & Contract Deadlines)

## Private Launch Budget & Spending Rules
- **Monthly Infrastructure Cap**: **$15,000 USD/month** for all production hosting, approved by the CFO.
- **Untouchable Reserve Rule**: Keep at least **$1,500 USD/month** unallocated for incident scaling and surprise egress bills.
- **Currency Exchange Reference**: `1 USD = 0.92 EUR` (use this rate whenever quoting in Euros; `1 USD = 150 JPY` for Yen).

## Hosting Vendor Quotes (Signed Holds Expiring Friday)
1. **Cluster Alpha (Shared Multi-Tenant GPU Pool — Quote #A-17)**:
   - **Region**: **us-west-1**
   - **Monthly Price**: **$8,200 USD/month** — shared multi-tenant hardware, 12-month commitment.
   - **Availability**: Ready immediately, no handover wait.
2. **Cluster Beta (Dedicated Single-Tenant H100 Nodes — Quote #B-04)**:
   - **Region**: **eu-central-1** (Frankfurt)
   - **Monthly Price**: **$12,800 USD/month** — dedicated single-tenant hardware, 12-month commitment.
   - **Hardware Handover**: Vendor hands the nodes over **Sunday at 7:00 PM CET**.
3. **Cluster Beta Early Handover (Add-On to Quote #B-04)**:
   - **One-Time Fee**: **$1,100 USD** to receive the same Beta nodes **Saturday at 9:00 AM CET** instead of Sunday evening.

## Customer Contract Deadlines & Penalties
- **St. Aurelia Health Go-Live Clause**: The patient portal must be **live by Monday 9:00 AM CET**. Missing it triggers a **$20,000 USD SLA penalty** and lets the hospital delay its renewal.
- **Klinik Nordhafen Pilot**: Starts two weeks after go-live; no fixed penalty.
