# P8 horizontal surface follow-up

Date: 2026-11-24. This addresses the user's observed floor patch seams after the
engineering pipeline acceptance; it does not claim that earlier acceptance proved
a globally smooth or safe floor.

## Implementation boundary

No upstream equivalent exists on local main. The opt-in helper lives in
`mjlab.asset_zoo.gaussian_collision_surfaces`; only optional provenance fields are
added to the existing primitive dataclass. The existing generator gains an explicit
`--merge-horizontal-surfaces` switch requiring auto mode. It runs before optional
budget selection. Default MJCF/source-ID NPZ output is unchanged byte for byte on
Artanim. Play/demo and their config tests are unchanged.

Candidate boxes qualify using source-point planarity, minimum XY span and a
near-vertical normal. They are grouped by semantic label and complete-link bounds
on existing box-top levels (not transitive chains of height differences). Grouped
source height spread must remain bounded. Existing free-side-anchored surfaces and
already consolidated surfaces are not overridden. The exact union of original
voxel-candidate XY footprints is retiled without hole filling. Rectangles share one
top level and finite thickness. Groups are rejected if they increase primitive
count, lose source mappings or intrude into explicit known-free/different-label
points. Full source-candidate membership is retained alongside the CSR mapping.

Default tolerances are level range 0.04 m, source p95 residual 0.06 m, slab
thickness 0.1 m and maximum tilt 5 degrees. These are engineering assumptions,
not semantic floor confidence; floors, shelves and other near-horizontal patches
can qualify. It does not infer unobserved holes or convert an entire room to one
unbounded plane. It preserves holes in candidate footprints, not necessarily every
hole in the original physical scene.

## Actual refinement and measurements

The first candidate aligned slabs to source-point median Z. Center-region
MuJoCo rays exposed a regression: 83 neighboring pairs developed >2 cm jumps,
versus zero before. That candidate is retained for audit at `horizontal-collision/`
and is **not recommended**. The implementation was corrected to align existing
collision top levels rather than interpreting Gaussian centers as surface tops.

The corrected output is `artifacts/p8-scenesplat/horizontal-collision-v2/`:

```bash
uv run scripts/tools/generate_3dgs_collision.py \
  src/mjlab/asset_zoo/3dgs/ArtanimMotionCaptureStudio/scene_transform_20p.ply \
  artifacts/p8-scenesplat/horizontal-collision-v2/collision.xml \
  --report artifacts/p8-scenesplat/horizontal-collision-v2/report.json \
  --primitive-mode auto --merge-horizontal-surfaces --evaluate-source-surface
```

- Original: 257 geoms; corrected: 255 (252 boxes, 3 cylinders).
- 27 resulting horizontal slabs share their respective group top levels.
- Exact mapped source-ID multiset retained, not merely the same count.
- Source-distance mean: 0.077869 -> 0.076717 m.
- Source-distance p95: 0.187743 -> 0.189030 m (slight regression, explicitly retained).
- Source-distance max: unchanged at 1.451725 m.

The center `[-2,2] x [-2,2]` was queried at 0.05 m spacing from z=1 m, with
MuJoCo downward rays, and the floor band restricted to [-0.4,0.2] m:

| Metric | Before | Corrected |
|---|---:|---:|
| Floor-band ray hits | 6,561 | 6,561 |
| Ray misses | 0 | 0 |
| Adjacent pairs with >2 cm jumps | 0 | 0 |
| Mean adjacent height difference | 0.198902 mm | 0.131136 mm |
| p95 adjacent height difference | 0.403937 mm | 0.230599 mm |

Thus visible patch edges in this crop did not prove physical floor cracks.
The corrected candidate reduces local height variation, but it does not remove
all patch boundaries or demonstrate dynamic walking stability. Center proxy-ray
statistics are not independent scene truth. Preview image: `support_heights.png`.
The original guarded collision is preserved and is not silently replaced.

The same 0.05 m grid was also extended over the entire proxy XY bounds. Floor-band
hits changed from 45,251 to 45,068 and misses from 73,047 to 73,230; neighboring
>2 cm jumps changed from 1,736 to 1,723, but p95 neighboring differences increased
from 2.533 to 3.477 mm. Retiling the voxel-candidate footprint union does not
preserve every extension of the old rotated padded boxes, so this is not evidence
of global non-regression. The candidate remains an opt-in preview, **not an
automatic replacement of the accepted guarded asset**. Full-scene dynamic contact
and physical support correctness are unverified. Further boundary-envelope handling
is required before promoting it globally.

This distinction is deliberate: fewer boxes or a smoother center crop is not enough
to conclude that a complete floor has been fixed.

Synthetic tests cover same-top alignment, reduced patch count, source-ID retention,
deterministic MJCF compilation, a finite pit and outer edge, separate 0.1 m levels,
explicit known-free rejection, semantic boundaries and ramp exclusion.

Validation: 46 focused tests pass; `make check`, `make docs`, and
`git diff --check` pass with dependency synchronization disabled for the existing
environment. Both original and corrected full proxies compile and run the same
2x2 falling-probe benchmark. Disabled-feature MJCF and source-ID NPZ match the
previous guarded baseline byte for byte. The only existing data-interface change
is optional primitive provenance metadata; no simulation-core or play/demo changes
were required. No claim of full-scene dynamic non-regression is made.
