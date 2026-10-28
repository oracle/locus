# Diagrams

Every diagram on the site is an SVG in this directory, inlined into its page by
the build hook:

```markdown
{{ locus_diagram agent-loop }}
```

Inlined, a diagram reads the page's `--tl-*` design tokens and follows the
light/dark toggle exactly. An `<img>` cannot see the page's theme, which is why
the old diagrams sat on a white card in dark mode. The hook also prefixes every
`id` with the diagram name, so two diagrams on one page never collide.

`agent-loop.svg` is the reference. Match it.

## Rules

1. **No inline styling.** No `fill`, `stroke`, `style`, `font-*` or `color`
   attributes. Every visual property comes from a class below, so the diagram
   themes itself. Geometry (`x`, `y`, `width`, `height`, `d`, `rx`, `points`)
   is fine.
2. **Accessible.** `<svg ... role="img" aria-labelledby="title desc">` with a
   short `<title id="title">` and a `<desc id="desc">` that says in prose what the
   diagram shows — the same facts a sighted reader gets.
3. **True to the SDK.** A diagram is a claim about how the code behaves. Draw
   it from the source, not from the old picture, and put the routing it
   depicts in a comment at the top (see `agent-loop.svg`).
4. **Colour is never the only cue.** A status node or edge also carries its
   word: allow, hold, deny, abstain.
5. **Pink means Locus or "look here".** Use `.primary` for the one node the
   diagram is about, not for every box.
6. **Nothing overlaps or clips.** Labels sit in clear space on a `label-bg`
   knockout, never on a node or another label; text stays inside its node and
   inside the viewBox. Avoid crossing edges — reroute instead.
7. **Stable facts only.** No model ids, version numbers or counts that change
   with a release.

## Vocabulary

| Element | Markup |
|---|---|
| Node | `<g class="node"><rect rx="12"/><text>Name</text><text class="sub">detail</text></g>` |
| The node the diagram is about | `g.node.primary` |
| Optional / conditional step | `g.node.optional` (dashed border) |
| Terminal result | `g.node.end` (inverted), `rx` = half the height |
| Decision outcome | `g.node.allow` · `.hold` · `.deny` · `.abstain` |
| Edge | `<path class="edge" marker-end="url(#ah)" d="…"/>`, add `.dashed` for conditional, `.allow`/`.hold`/`.deny` to colour an outcome path |
| Edge label | `<rect class="label-bg" …/>` then `<text class="label">` centred on it |
| Edge label on a lane | the same, with `class="label-bg on-lane"` so the knockout matches the band |
| Datastore | `<g class="store"><path d="cylinder"/></g>` |
| Decision | `<g class="decision node"><polygon/><text>Name</text><text class="sub">question?</text></g>` — `decision` styles the polygon, `node` styles and centres the text; a bare question is a single `text.sub` at the polygon's centre |
| Grouping band | `<g class="lane"><rect/><text class="lane-title">NAME</text></g>` |
| Footnote | `<text class="note">` |

One arrowhead marker, `id="ah"`, with `<path class="arrowhead" d="M0,0 L10,5 L0,10 z"/>`.

Node text is centred (`text-anchor: middle` comes from the class): put the
name at the node's vertical centre − 10 and the `sub` line at centre + 11.
Size nodes so the longest line has at least 12 px of padding each side.

## Check before committing

```bash
python render_check.py docs/diagrams/NAME.svg /tmp/NAME
```

renders it in light, dark and at phone width, and fails on inline styling,
unknown classes, a missing title/description, or any clipped, overlapping or
overflowing text.
