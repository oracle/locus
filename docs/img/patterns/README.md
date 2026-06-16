# Multi-agent pattern diagrams

The seven in-process multi-agent + A2A pattern diagrams are now authored in
**[draw.io](https://app.diagrams.net/) (diagrams.net)** for editing,
and exported to SVG for the docs site.

## Files

| | |
|---|---|
| **`locus-patterns.drawio`** | Single source-of-truth file. Open in diagrams.net to edit. Has seven tabs — Composition, Orchestrator, Swarm, Handoff, StateGraph, Functional, A2A — each rendered identically to one of the SVGs below. |
| `composition.svg` | Rendered Composition diagram. Embedded in `docs/concepts/multi-agent/composition.md`. |
| `orchestrator.svg` | Rendered Orchestrator diagram. |
| `swarm.svg` | Rendered Swarm diagram. |
| `handoff.svg` | Rendered Handoff diagram. |
| `graph.svg` | Rendered StateGraph diagram. |
| `functional.svg` | Rendered Functional API diagram. |
| `a2a.svg` | Rendered A2A diagram. |

## Edit workflow

1. Open <https://app.diagrams.net/> and pick **Open existing diagram** → upload
   `locus-patterns.drawio`.
2. Click the tab for the pattern you want to edit (Composition, Swarm, etc.).
3. Edit the shapes / labels / colours in the diagram.net GUI.
4. **`File → Save`** — overwrites the `.drawio` source.
5. **`File → Export as → SVG…`** — uncheck *"Include a copy of my diagram"*
   if you want a smaller file. Save as `<pattern>.svg` in this directory.
6. Commit both the `.drawio` source and the regenerated `.svg`.

## Colour palette (matches the locus brand)

These come from the Oracle FY26 PowerPoint template and the locus
brand sheet:

| Use | Hex |
|---|---|
| Think / source / structure | `#04536F` (deep teal · accent1) |
| Execute / primary action / final | `#C74634` (Oracle red · accent3) |
| Reflect / data plane / sage cards | `#89B2B0` (sage teal · accent5) |
| Terminate / shared state / decision | `#F0CC71` (sand · accent4) |
| Mauve / dashed result-flow arrows | `#6C3F49` (mauve · accent2) |
| Card text on dark cards | `#FFFFFF` |
| Card text on light cards | `#1F2828` / `#3A2A0F` |
| Hairlines, default text | `#2A2F2F` (dk1) |

The dashed-mauve arrow style is the convention for **derived /
result data flowing back** (the merge step in Composition's parallel
mode, the responses returning to the Coordinator in Orchestrator,
the gather step in Functional). The solid-Oracle-red arrow is for
**primary cross-boundary connections** (handoff, A2A wire).

## Why draw.io

- **Open format** — the `.drawio` file is XML, diff-friendly under
  git.
- **Editable in browser** — no install needed; <https://app.diagrams.net/>.
- **Exports SVG** that renders crisply at any size.
- **Works offline** — the desktop app at <https://github.com/jgraph/drawio-desktop>
  reads the same files.
