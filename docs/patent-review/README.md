# MTR-KV patent-review handoff (technical, not legal advice)

Prepared 18 October 2026 against the frozen 22 September 2026 architecture and the current working tree. **Status: technical disclosure and preliminary evidence pack; not a filing-ready specification, novelty opinion, freedom-to-operate opinion, or proof of a VLM/GPU speedup.** A qualified patent agent must settle claim scope, inventorship, disclosure history, prior-art search, and jurisdiction-specific filing strategy.

## Contents

- [Technical specification working draft](draft-technical-specification.md): counsel-oriented system/transaction description, reproducible software method, proposed alternatives, and drawing package; not final claims.
- [Technical disclosure addendum](technical-disclosure-addendum.md): embodiment and drawing instructions, clearly distinguishing implementation from proposal.
- [Claim-element evidence map](claim-evidence-map.md): proposed technical elements versus code, tests, and missing proof.
- [Preliminary prior-art landscape](prior-art-landscape.md): verified patent publications and papers, overlap and unresolved questions.
- [Validation protocol](validation-protocol.md): repeatable real-video capture and acceptance gates.
- [Counsel handoff](counsel-handoff.md): decisions and facts the inventors must supply.

The original `docs/MTR_KV_Frozen_Master_Architecture_and_Prototype_Specification_FINAL_v1_0.docx` is the inventors' architecture disclosure. It is **not** treated as evidence that all described hardware/VLM embodiments exist. The `results/simulation.json` artifact contains synthetic simulation, not real-hardware measurements. No representative real-video dataset was available when this pack was prepared, so no real-workload performance result is asserted here.

Fresh synthetic control traces for semantic rescue (`results/patent-review/T03-simulation.json`), scene transition (`results/patent-review/T04-simulation.json`), and metadata fallback (`results/patent-review/T08-simulation.json`) were generated from the reference configuration. Their frame CSVs and `T03-report.md` sit beside them. They support behavioral review only; they are **not** a real-video performance study.

## Readiness gate

1. **Documentary:** inventor review of this pack and the frozen disclosure, dates of any public demonstrations/publications, then independent patent-agent search and drafting.
2. **Technical:** acquire a licensable representative video corpus, run the hashed evidence protocol, preserve raw traces, and audit quality/latency tradeoffs.
3. **Broader claims:** only after implementation and validation of a selected VLM, cross-frame reuse, and the hardware/device path should the corresponding technical effects be asserted.

For India, [Section 10 of the Patents Act](https://www.ipindia.gov.in/acts/patent-act-1970/section-10) requires a complete specification to describe operation and the known best method and to define claims. The [2025 CRI Guidelines](https://ipindia.gov.in/frontend/pdf/patents/guidelines/GUIDELINES%20FOR%20EXAMINATION%20OF%20COMPUTER%20RELATED%20INVENTIONS%20%28CRIs%29%20-%202025.pdf) discuss technical effect for computer-related inventions; they do not make this particular project patentable. Consult counsel **before public disclosure**; [WIPO explains](https://www.wipo.int/en/web/patents/faq_patents) the novelty risk of pre-filing disclosure. No legal conclusion is drawn from these links.
