# Label definitions

Each file is a ready-to-use **optimizable prompt** (the section APE rewrites).
Two granularities are provided per task:

| Task | Simple | Fine-grained |
|------|--------|--------------|
| Security vs. Non-security | `simple/security.txt` | `finegrained/security.txt` |
| Functional vs. Non-functional | `simple/functional.txt` | `finegrained/functional.txt` |
| Quality vs. Non-quality | `simple/quality.txt` | `finegrained/quality.txt` |

- **Simple** definitions are one or two sentences per class — a minimal seed.
- **Fine-grained** definitions are the full, literature-grounded descriptions
  (Nuseibeh; ISO/IEC/IEEE 29148; Firesmith; Glinz; ISO 25010-derived quality
  taxonomies). Use these as the strong baseline / starting point for APE.

Pass one to the runner with `--definitions`, or paste it into the GUI's
"optimisable prompt" box.

## Sources
- Security: *A Framework for Security Requirements Engineering* (Nuseibeh et al.);
  ISO/IEC/IEEE 29148:2018; *Software Engineering for Security: a Roadmap*;
  *Specifying Reusable Security Requirements* (Firesmith).
- Functional / Non-functional: Glinz, *On Non-Functional Requirements* (RE'07).
- Quality / Non-quality: ISO/IEC 25010-derived product- and quality-in-use models.
