---
name: hua
description: >
  Terminal-based reader for classical Chinese poetry and prose.
  Core Scenario: When the user wants to read and appreciate classical Chinese literature (Shi Jing, Tang Poetry, etc.) in the terminal.
license: MIT
---

# hua - Classical Chinese Literature Reader

The `hua` module is a specialized terminal reader designed for Chinese classical literature. It contains a curated collection of poetry and prose ranging from the pre-Qin period to the Qing Dynasty.

## When to Activate
- When the user wants to read classical Chinese poetry (Tang, Song, etc.) in the terminal.
- When exploring philosophical works like the Analects (Lun Yu) or Mencius (Meng Zi).
- When seeking an interactive interface to browse classic Chinese texts.

## Core Principles & Rules
- **Curated Collections**: Organized into Classics, Confucian works, Poetry, and Prose.
- **Interactive Experience**: Optimized for immersive reading within the terminal environment.

## Patterns & Examples

### Browse All
```bash
# Open the interactive interface to explore all collections
x hua
```

### Read Specific Work
```bash
# Read selections from Tang Poetry 300
x hua ts
# Read the Classic of Poetry (Shi Jing)
x hua sj
```

## Checklist
- [ ] Confirm the specific work or dynasty the user is interested in.
- [ ] Verify if an interactive or direct reading experience is preferred.
