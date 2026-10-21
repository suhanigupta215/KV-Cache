# Questions and materials for a patent agent

This document is a handoff checklist, not a request for the agent to adopt any proposed conclusion.

## Facts the inventors must confirm

- Who conceived each technical element, on what date, and with what contemporaneous notes? The frozen document lists three contributors; names on that document alone do not determine legal inventorship.
- Has any part been presented to a professor, posted online, submitted as a thesis/paper, shown in a public demo, or shared without confidentiality? Record exact dates, recipients, links, and copies. Do not assume a conversation or repository is private without checking its access history.
- Is the repository or any related patent application already publicly accessible or filed? If filed, provide application number and priority documents. The frozen document's date is **not** automatically a patent priority date.
- Which embodiments are actually invented and enabled beyond the Python code: hardware interface, FIFO, page-table writer, VLM hooks, cross-frame reuse, prefetch, T3 reconstruction? Supply diagrams, source, and test records for each.
- What is the best known embodiment today, including decoder API, tier hardware, model, thresholds, page size, and failure handling? Identify choices based on evidence rather than idealized architecture.
- Who owns the rights under employment, university, sponsorship, model licenses, data licenses, and contributor agreements?

## Questions for professional analysis

1. Which candidate technical relationships, if any, remain novel and inventive over the full CodecSight/CoPE-VideoLM papers, PagedAttention, the listed K/V patent families, and the decoder-memory art? Compare **limitation by limitation** and search additional references.
2. Under the intended jurisdictions, what narrower apparatus/method claims are supportable by an enabling description without implying that unbuilt hardware or VLM embodiments were demonstrated?
3. How should India Section 3(k) / CRI technical-effect issues be addressed with truthful structural disclosure and evidence, rather than merely labeling software a “hardware controller”? See the [Indian Patent Office CRI guidance](https://ipindia.gov.in/frontend/pdf/patents/guidelines/GUIDELINES%20FOR%20EXAMINATION%20OF%20COMPUTER%20RELATED%20INVENTIONS%20%28CRIs%29%20-%202025.pdf).
4. Is a provisional filing appropriate before further public disclosure, and what must be included now to support later claims? What should not be publicly presented until filing strategy is settled?
5. Which drawings, alternatives, failure modes, and best-mode details are required for a complete specification under [Section 10](https://www.ipindia.gov.in/acts/patent-act-1970/section-10)?

## Deliver to counsel

The frozen master DOCX; this review pack; current source tree and test report; a clean revision/snapshot identifying the exact code reviewed; `results/simulation.json` labeled synthetic; any future `mtr_kv evidence` manifest and raw traces; prior invention disclosures and publication history; all inventor notebooks and hardware/VLM design records. **Do not replace missing materials with guessed results or dates.**
