# Data layout, tasks, and dataset sources

Every experiment reads tasks from `data/tasks/<bucket>/tasks.jsonl` — one JSON line per task, normalized through a single frozen [`TaskDefinition`](../experiments/common/tasks.py) dataclass. The jsonls are committed; most reference images are gitignored and re-created by per-source extractors under `data/datasets/`.

```
data/
├── solid_colors/           # CHECKED IN: 8 solid-color PNGs (solid_color axis)
├── style_references/       # CHECKED IN: 36 style-axis refs (18 illustrations + 18 real-photo counterparts)
│   ├── fictional/          #   illustrations
│   └── real/               #   real-photo counterparts
├── tasks/                  # CHECKED IN: jsonl source-of-truth per bucket
│   ├── add/tasks.jsonl
│   ├── remove/tasks.jsonl
│   ├── style/tasks.jsonl
│   ├── dreambench_humans/tasks.jsonl
│   ├── dreambench_humans_shared/tasks.jsonl
│   ├── customize/          # not a task bucket — holds shared resources
│   │   ├── images/         # gitignored except 10 dreambench-humans jpgs
│   │   ├── _style_prompts.yaml
│   │   └── _dreambench_human_shared_prompts.yaml
│   ├── manual/tasks.jsonl
│   └── solid_color/tasks.jsonl
└── datasets/               # raw downloads + per-source download.py / extract.py (raw/ subtrees gitignored)
    ├── _vlm_tasks.py       # shared edit-proposal prompt + validate + row-builder
    ├── _anchor.py          # anchor_to_reference() — instruction rewriting for customize
    ├── _image_utils.py     # center_crop_to_multiple()
    ├── sun397/             # download.py + extract.py
    └── dreambench_plus/    # download.py + extract.py
```

## Sources

Every TaskDefinition row carries a `source` field naming where the image came from.

| `source` | image origin | instruction origin | buckets (counts) |
|---|---|---|---|
| `sun397` | SUN397, 2 random JPGs per category × 397 categories | Claude Opus 4.7 (ban-list prompt) | add (789), remove (726) |
| `dreambench_plus` | DreamBench++, 1 image per real-human subject | dataset captions + 9 individualized prompts per subject (90); 5 shared prompts across all subjects (50) | dreambench_humans (90), dreambench_humans_shared (50) |
| `property_manual` | hand-built (refs in `data/style_references/fictional/`) | hand-written | style (450 = 18 subjects × 5 prompts × 5 seeds) |
| `manual` | hand-built (refs in `data/style_references/real/`) | hand-written | manual (450 real-photo style analogues used as i2i→i2i targets) |
| `solid_color` | synthetic 8 colors × 8 objects | hand-written, anchored on `solid_<color>` real_ref | solid_color (320 = 8 × 8 × 5 seeds) |

Bucket totals (2,875 rows total across `tasks/*/tasks.jsonl`):

| bucket | total | notes |
|---|---:|---|
| add | 789 | all in judge CSVs |
| remove | 726 | all in judge CSVs |
| style | 450 | property_manual; 18 subjects × 5 prompts × 5 seeds |
| dreambench_humans | 90 | 10 real-human subjects × 9 individualized prompts (KO + T2I Lens) |
| dreambench_humans_shared | 50 | same 10 subjects × 5 shared prompts (I2I-to-I2I pair source) |
| manual | 450 | real-photo i2i→i2i targets |
| solid_color | 320 | 8 × 8 × 5 |

Human customization in the paper (140) = `dreambench_humans` + `dreambench_humans_shared`.

## TaskDefinition

```python
@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    edit_type: Literal["add", "remove", "customize"]
    source: Literal["dreambench_plus", "manual", "solid_color", "property_manual", "sun397"]
    instruction: str
    source_image_path: str | None
    source_caption: str | None   # for synth-ref scenes: prompt for ref generation
    ref_seed: int | None
    noise_seed: int | None
    real_ref_name: str | None    # filename under <real_ref_dir>
    real_ref_dir: str | None     # set together with real_ref_name
    height: int = 1024
    width: int = 1024
    metadata: dict = field(default_factory=dict)
```

Invariants (asserted in `__post_init__`): at least one of `source_image_path` / `real_ref_name` / `ref_seed` must be set; `source_image_path` and `real_ref_name` are mutually exclusive; `ref_seed != noise_seed` when both are set; `height` / `width` are multiples of 16 (FLUX VAE constraint).

## Reference image dispatch

[`experiments/common/baselines.py`](../experiments/common/baselines.py) resolves the i2i reference from one of three sources, in priority order:

1. `real_ref_name` → `<real_ref_dir>/<name>.{png,jpg,jpeg}` (style refs live under `data/style_references/{fictional,real}/`; solid colors under `data/solid_colors/`).
2. `source_image_path` → load disk image, resize to `(width, height)`.
3. `ref_seed` → generate via `model.generate(source_caption, seed=ref_seed)`.

## Loader API

```python
from experiments.common.tasks import load_tasks, get_task

tasks = load_tasks("remove")               # all 726 remove tasks
tasks = load_tasks("add", limit=50)        # first 50
tasks = load_tasks("add", source="sun397") # filter by source
task = get_task("manual_solid_yellow_couch")  # by id, searches every bucket
```

CLI wrappers in [`experiments/common/cli.py`](../experiments/common/cli.py) (`add_task_selection` / `resolve_tasks`) expose `--task-id`, `--bucket`, and `--edit-type` for entry scripts.

## Reproducing the data

Most per-task images are gitignored. The committed jsonls reference them by path; regenerate locally with each extractor.

| Source | Setup | Extract | Images land at |
|---|---|---|---|
| `dreambench_plus` | `huggingface_hub.snapshot_download("yuangpeng/dreambench_plus")`, unzip into `data/datasets/dreambench_plus/raw/` (Apache-2.0); [`download.py`](datasets/dreambench_plus/download.py) prints exact instructions. | `uv run python -m data.datasets.dreambench_plus.extract` | `data/tasks/customize/images/customize_dreambench_plus_*.jpg` |
| `sun397` | Download from <https://vision.princeton.edu/projects/2010/SUN/> (research-only; no scriptable URL). On CSAIL the Torralba NFS mount holds it at `/data/vision/torralba/datasets/SUN397/SUN397`. | `uv run python -m data.datasets.sun397.prepare_images --root <path-to-SUN397>` | `data/tasks/{add,remove}/images/sun397_*.jpg` |

The 10 DreamBench++ real-human reference jpgs (`customize_dreambench_plus_live_subject_human_*.jpg`) are committed under `data/tasks/customize/images/`, so both the `dreambench_humans` and `dreambench_humans_shared` families run without the DreamBench++ download. (`customize/` no longer contains a `tasks.jsonl`; it survives as the shared-resources directory for the two dreambench buckets — images, prompt yamls, extract logs.)

The 789 add + 726 remove SUN397 task instructions are committed in `data/tasks/{add,remove}/tasks.jsonl`; the script that authored them lives at [`extract_instructions.py`](datasets/sun397/extract_instructions.py) and is only needed to extend the task set (requires `ANTHROPIC_API_KEY`).

Every extractor center-crops to a multiple of 16, validates each row by constructing a `TaskDefinition`, and skips images whose task_ids are already present in the JSONL (re-runs incur zero VLM API cost). The sun397 extractor additionally applies [`utils.flux2_klein.effective_ref_dims`](../utils/flux2_klein.py) — lanczos downscale to ≤1024² area, multiple-of-16 floor — mirroring Flux2KleinPipeline's internal resize so saved JPG dims match the runner's effective dims. SUN397 instructions come from one Anthropic call per image returning `{add_object, remove_object}`; the system prompt bans scene-revealing names (e.g. "lava plume" for a volcano photo) plus a fixed stock-object ban list. 789 add tasks → 175 distinct subjects, top subject ≤7.2%.

The shared async Anthropic client lives at [`utils/vlm.py`](../utils/vlm.py); the edit-proposal prompt + validation + row-builder live at [`datasets/_vlm_tasks.py`](datasets/_vlm_tasks.py).

## Conventions

- **Customize anchor.** All customize instructions are anchored on **"in this image"** (or **"as this image"** for property axes) — never "the reference image". Shared anchor in [`datasets/_anchor.py`](datasets/_anchor.py).
- **Noise seed.** Every task gets `noise_seed = task_seed(task_id)` where `task_seed = zlib.crc32(task_id.encode())` ([`tasks/_seed.py`](tasks/_seed.py)) — deterministic, reproducible from the task_id alone.
- **Property-axis replicates.** `solid_color` and the `style` axis of `property_manual` are expanded to 5 replicates per concept (`<id>_s0` … `<id>_s4`). Each replicate's seed is `task_seed(<new_id>)`, giving 5 distinct i2i generations and (via `+1` seed offset in i2i-to-T2I-unconditional) 5 distinct t2i generations per concept.
- **i2i→i2i pair invariants.** Pairs assert `source.instruction == target.instruction`, `source.noise_seed == target.noise_seed`, and `_ref_key(source) != _ref_key(target)`. Pair files (`<source_id>\t<target_id>`) are emitted by [`build_pairs_color.py`](../experiments/i2i_to_i2i_patching/build_pairs_color.py), [`build_pairs_style.py`](../experiments/i2i_to_i2i_patching/build_pairs_style.py), and [`build_pairs_dreambench_humans.py`](../experiments/i2i_to_i2i_patching/build_pairs_dreambench_humans.py). For each style subject the pair is `customize_property_style_<subject>_<action>_sN` ↔ `manual_<subject>_real_sN`: same instruction (manual_real adopts the standalone style instruction), same seed, different ref.

## Example JSONL row

```json
{"task_id":"solid_red_couch_s0","edit_type":"customize","source":"solid_color",
 "instruction":"draw a couch in this color",
 "source_image_path":null,"source_caption":null,"ref_seed":null,
 "noise_seed":1635624438,"real_ref_name":"solid_red",
 "height":1024,"width":1024,"metadata":{}}
```
