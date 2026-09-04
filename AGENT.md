# AGENT.md — therobotgguf

Ground-up transformer/GGUF design & tooling informed by Noizu AI notes (arch/, convert/, runpod/, extraction pipeline docs). Monorepo role: AI tooling, couples to the Libs/ai (genai) family. Status: docs/architecture-heavy, no build system at root.

## Universal rules (monorepo policy)

- **Trinity Protocol (REQUIRED)**: substantive responses follow Orientation (assumption table, minds-eye, mermaid plan) → Friction (WEDGE/SHADOW/CRITIC) → Response + meta-review. Full text: monorepo `protocols/the-trinity-protocol.md`.
- **No shell in main thread** — delegate lookups/builds/greps to tasker subagents; batch and summarize.
- **Worktree workflow (REQUIRED)**: all work on worktrees; integration-testing consolidation branches `epic.<group>` fork from active `develop`; feature branches merge into their parent epic via squash-PR (provenance); a fully-passing epic becomes one PR for the group. See monorepo CLAUDE.md "Git Trees — Worktree Workflow".

Monorepo-wide ops (secrets/dc, terraform, submodules, tiers): see `../../../CLAUDE.md` at the trl-infra root.
