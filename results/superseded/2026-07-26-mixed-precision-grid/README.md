# Superseded AMP-core and reference result export

This is the tracked export subset from campaign `fresh34-v100-20260724`, run
from `dev` at `2fb24e08b903313fe097e5c559218a2201c06de1`. The eight exported f10
core cells used `16-mixed` and detector SHA256
`4fd1bfe88861cc676dd67b2092e379fbcf401dd9c1d42fb09e81a84b9cdbe2f8`.
The R2/R3 exports retained their independent published precision recipes;
they are superseded because the owner requested both references fresh, not
because they shared the core AMP recipe.

The tracked subset contains eight f10 core exports, R2/R3, and their grid
summary. The full stopped runtime state was 10 complete, 6 interrupted,
1 failed, and 17 unstarted; it remains in the ignored local archive at
`runs/archive/superseded_amp_fresh34_20260726T211204Z/`.

These files are provenance only. Do not combine them with the fresh `32-true`
core grid or publish their summary as the active result table. The replacement
campaign reruns all 32 core cells and both references from empty canonical
namespaces. P3.6 diagnostics and hardware-throughput probes remain outside
this directory because they are explicitly labeled gates, not reportable grid
cells.
