# Liam — Private Insider Notes (Migration Timings, Benchmarks & Integration Internals)

## Personal Migration Dry-Run Logs
- **Full Migration to Cluster Beta (Timed Dry Run Last Week)**:
  - Snapshotting the production database, replicating 4.2 TB into eu-central-1, re-indexing, and warming the caches takes **14 hours** end to end. It runs unattended except for a **checkpoint every 4 hours** that an engineer must approve.
- **Full Migration to Cluster Alpha (Timed Dry Run)**:
  - Takes **6 hours** end to end because the existing staging replica already lives in us-west-1.
- **Rollback Window**: Rolling back either migration takes **3 hours**.

## Latency Benchmarks for EU Users
1. **Cluster Alpha (us-west-1)**: p99 latency of **180 ms** for European users; uses shared memory buffers across tenants.
2. **Cluster Beta (eu-central-1)**: p99 latency of **22 ms** for European users; supports dedicated NVMe encryption at rest.

## Integration Holds & Insider Data-Handling Notes
1. **VectorFlow Managed Semantic Search (Vendor Trial Hold)**:
   - **How It Handles Data & Insider Warning (MUST MENTION WHEN DISCUSSING VECTORFLOW)**: VectorFlow copies every indexed document, including patient-portal message text, into a **shared multi-tenant index hosted in us-east-1**, and their sales engineer confirmed they **cannot guarantee EU-only storage** or sign a data processing agreement before launch.
   - **Integration Effort**: **2 days**; trial pricing is free for the first 3 months.
2. **Self-Hosted pgvector on Cluster Beta (Internal Option)**:
   - **How It Handles Data (MUST MENTION WHEN DISCUSSING PGVECTOR)**: Runs **inside our own eu-central-1 database** on dedicated hardware; **no data leaves our cluster** and no new sub-processor is involved.
   - **Integration Effort**: **5 days**; search quality was within 3% of VectorFlow in my benchmark.

## Low-Traffic Windows
- **Patient-Portal Traffic**: Lowest between **Sunday 2:00 AM – 6:00 AM CET**; peaks Monday **8:00 AM – 11:00 AM CET** when clinics open.
