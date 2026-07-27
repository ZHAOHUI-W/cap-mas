# Research Sources and Design Traceability

This document separates literature-grounded inspiration from CAP-MAS claims that require new experiments. It is not a substitute for a citation-verified related-work section.

| Source | Design element used | What it supports | What it does not establish for CAP-MAS |
| --- | --- | --- | --- |
| CaP-X, arXiv:2603.22435 | CAP-X-compatible code execution, multi-turn feedback, visual differencing, skill-library baseline | Code-as-Policy evaluation and test-time interaction are a valid comparison base | It does not establish contract-driven multi-agent coordination |
| CaP-RL implementation path in CAP-X | GRPO post-training and simulator reward interfaces | CAP-X RL remains an available training/baseline path | It does not mean CAP-MAS online runtime performs RL |
| Playful Agentic Robot Learning, arXiv:2606.19419 | Robot-agent teams, exploratory skill acquisition, verification, failure diagnosis, skill distillation | Play-time skill acquisition and verification are relevant to robot learning | Its reported gains cannot be transferred directly to CAP-MAS's contract runtime |
| MetaGen, arXiv:2601.19290 | Query-conditioned roles, sparse DAG topology, inference-time role/topology adaptation, cost-aware interaction | Roles and communication structure can be treated as adaptive runtime objects | Text reasoning results do not prove physical-state consistency or real-time control |
| ENPIRE, arXiv:2606.19980 | Reset-execute-verify-refine loop | Explicit verification and iterative improvement are useful design motifs | Its evolution loop does not by itself solve stale physical actions |
| ReVeal, arXiv:2506.11442 | Multi-turn generation-verification and turn-level attribution | Verification and feedback attribution are relevant | Turn-level reasoning results are not robot-control evidence |
| SEEA-R1, arXiv:2506.21669 | Dense intermediate feedback | Intermediate signals may help long-horizon credit assignment | Reward design does not replace state contracts or safety checks |
| MemSkill, arXiv:2602.02474 | Controller, executor, and designer separation | Skill memory and designer/executor separation are relevant | Memory reuse alone does not guarantee OOD transfer or no regression |
| SkillRL, arXiv:2602.08234 | Hierarchical SkillBank and adaptive retrieval | Skill hierarchy and retrieval are useful extensions | It does not establish CAP-MAS's real-time perception protocol |
| CoEvoSkills, arXiv:2604.01687 | Generator/verifier co-evolution | Candidate generation and surrogate verification are useful for quarantine testing | Co-evolution can introduce instability without CAP-MAS promotion gates |

## Evidence status

- Local PDFs are available under the CAP-X workspace at ../cap-x/docs/papers/.
- Primary paper details, reported numbers, and venue status must be verified before inclusion in a manuscript.
- CAP-MAS hypotheses H1-H5 are proposed research claims, not literature findings.
- Real-time rates in real-time-perception.md are engineering targets for the prototype, not claims that the cited papers meet those rates.
