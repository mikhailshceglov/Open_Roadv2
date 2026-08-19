# Objectomaly licensing status

## Current status

The pinned Objectomaly upstream commit
`66d2ad2a1b02d79389f4265d9d1d99ab6412324f` contains no project-level
`LICENSE`. Its README explicitly says that a license still needs to be
selected. A public GitHub repository and a link from a paper do not by
themselves grant permission to copy, modify, or redistribute its code.

RAAS therefore keeps Objectomaly as an unmodified git submodule and keeps all
new bridge code on the RAAS side. This preserves provenance but does not
replace permission from the copyright holders.

## What counts as confirmation

Use one of these forms:

1. the authors add a recognized `LICENSE` file to the official repository;
2. the repository owner/copyright holders provide written permission that
   explicitly covers the intended academic use, modification and
   redistribution of the integration;
3. the authors identify an existing license that applies to the entire
   repository and clarify this in an official repository issue or README.

Record the upstream commit containing the license or retain a stable link to
the author's written response. Ask the team/legal owner whether additional
review is required before publishing code, weights, containers or derived
artifacts.

The following are **not** sufficient: the repository being public, the paper
linking to it, the ability to clone/fork it, or the absence of an explicit
prohibition.

## Suggested GitHub issue or email

**Subject:** License clarification for academic Objectomaly integration

Hello,

We are evaluating Objectomaly in an academic road-anomaly segmentation
project and would like to integrate the official implementation with RAAS.
We currently reference your repository as an unmodified pinned git submodule
and keep our adapter code separate.

The repository does not currently contain a project-level LICENSE file.
Could you please clarify which license applies to the Objectomaly source code?
In particular, may we use, modify, and redistribute an integration for
academic research, with attribution and a link to the official repository?

If possible, adding the applicable LICENSE file to the repository would make
the permission clear and reproducible for downstream users.

Thank you.

## Until confirmation arrives

- local/internal evaluation may proceed subject to the team's policy;
- do not vendor or modify Objectomaly files;
- do not publish a combined source archive or container containing Objectomaly;
- do not claim that Objectomaly is open-source;
- keep the exact upstream commit and attribution in every experiment manifest.
