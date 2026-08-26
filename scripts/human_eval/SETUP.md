# Running the human evaluation on Amazon Mechanical Turk

End-to-end operational runbook. [README.md](README.md) explains *what* the study
measures and why it is designed the way it is; this file is *how to actually run
it*, including every gotcha that cost us time the first time through.

Nothing here is MTurk-specific until [step 4](#4-build-the-six-mturk-projects).
Steps 1–3 produce a platform-agnostic study (a manifest, 534 images, six HTML
layouts, six input CSVs) that also drops into Prolific, CloudResearch or a
self-hosted Turkle — see [Alternatives](#alternatives-to-mturk).

---

## At a glance

| | |
|---|---|
| Tasks (paper cells) | 120 |
| Items (questions asked) | 294 |
| Images | 534 |
| HITs | 17 |
| Assignments | 51 (3 workers × 17 HITs) |
| Judgements | 882 |
| Reward | $1.00 per assignment |
| **Total cost** | **$61.20** ($51.00 rewards + 20% MTurk fee) |

---

## 1. Accounts

> **MTurk closed to new customers on 2026-07-30.** Existing accounts are
> unaffected, but you can no longer create one. If you don't have a requester
> account already, skip to [Alternatives](#alternatives-to-mturk).

Four registrations, all free:

1. **AWS account** — <https://portal.aws.amazon.com/billing/signup>. Note the
   account ID.
2. **Requester** — <https://requester.mturk.com> → *Create an account*. Uses
   your Amazon retail login.
3. **Requester sandbox** — <https://requestersandbox.mturk.com>. A separate
   registration, not a mode toggle. The same email is fine.
4. **Worker sandbox** — <https://workersandbox.mturk.com>. Needed to test your
   own HITs. Same email is fine *in the sandbox only*.

Then link AWS to MTurk at <https://requester.mturk.com/developer> →
*Link your AWS Account*, and repeat at
<https://requestersandbox.mturk.com/developer>.

You do **not** need IAM access keys for MTurk itself — 17 HITs is a web-UI job.
You do need them for S3 in step 3.

### The credit limit will block you

**New requester accounts start with a monthly credit limit of zero.** Publishing
anything fails with:

> *You have exceeded your monthly credit limit, please contact us to request a
> limit increase on your AWS MTurk account for your expected usage.*

This is expected, not a misconfiguration. Request an increase at
<https://www.mturk.com/contact-us> and
<https://support.aws.amazon.com/#/contacts/aws-mechanical-turk>. Increases are
handled case by case with no SLA — reports range from days to weeks, and some
go unanswered. **Request it before you need it.**

Support will ask for three things; answer them concretely:

| Question | This study |
|---|---|
| Number of workers / assignments | 51 assignments over 17 HITs, 3 per HIT, so at most 51 distinct workers |
| Payment per task | $1.00 per assignment — 16–20 comparisons, 3–4 min, ≈$15–20/hr |
| Project duration | One batch, ~1 week |

Ask for ~$200/month rather than the exact $61.20, so a re-run after a layout fix
doesn't put you back in the queue. Mention the effective hourly rate and the
academic context up front; underpayment and spam are the visible concerns.

The **sandbox is unaffected by the credit limit** (stock $10,000 of fake money),
so you can complete steps 2–5 while the request is pending.

---

## 2. Build the study

```bash
uv run python -m scripts.human_eval.select_subset \
    --base-url https://<your-bucket>.s3.amazonaws.com/images
```

Writes to `results_v4/human_eval/`:

| File | Tracked in git? |
|---|---|
| `manifest.csv` — 294 items, the join key for everything downstream | yes |
| `mturk_input_<group>.csv` × 6 | yes |
| `hit_<group>.html` × 6 | yes |
| `images/` — 534 web-sized JPEGs (~26 MB) | no, regenerated |

Requires the paper-scale sweeps under `results_v4/attention_knockout/` and
`results_v4/i2i_to_i2i_patching/` (plus `results_v4/i2i_to_unconditional/` for
the human check arm). Selection is deterministic — sha256-derived seeds — so the
same inputs always give the same 294 items.

Re-run it any time you change `--base-url`; image filenames are content-derived,
so nothing already uploaded goes stale.

---

## 3. Host the images

MTurk hosts nothing. Each HIT is `<img src="https://…">`, so the images need a
public URL.

```bash
uv tool install awscli          # if you don't have it

# An IAM user with AmazonS3FullAccess; do not use the AWS root account.
aws configure                   # region us-east-1
aws sts get-caller-identity     # sanity check

BUCKET=your-unique-bucket-name  # globally unique, lowercase, no underscores
aws s3api create-bucket --bucket "$BUCKET" --region us-east-1
```

Use `us-east-1` so the URL is the short `https://$BUCKET.s3.amazonaws.com/…`;
other regions need `https://$BUCKET.s3.<region>.amazonaws.com/…` and a mismatch
is a confusing 301.

The bucket name appears in every URL a worker sees. Don't name it after the
experiment.

### Public read comes from a bucket policy, not an ACL

`--acl public-read` **fails** on any bucket created after April 2023: new buckets
have Object Ownership = *bucket owner enforced*, which disables ACLs entirely and
returns `AccessControlListNotSupported`. Two gates must be opened instead:

```bash
aws s3api delete-public-access-block --bucket "$BUCKET"

aws s3api put-bucket-policy --bucket "$BUCKET" --policy '{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "PublicReadImages",
    "Effect": "Allow",
    "Principal": "*",
    "Action": "s3:GetObject",
    "Resource": "arn:aws:s3:::'"$BUCKET"'/images/*"
  }]
}'

aws s3 sync results_v4/human_eval/images "s3://$BUCKET/images" \
    --content-type image/jpeg
```

The policy is scoped to `images/*`, so nothing else in the bucket is exposed.
`--content-type` matters: without it S3 serves `binary/octet-stream` and some
browsers download instead of rendering.

### Verify from a logged-out browser

```bash
aws s3 ls "s3://$BUCKET/images/" | wc -l     # must be 534
curl -sI "https://$BUCKET.s3.amazonaws.com/images/$(ls results_v4/human_eval/images | head -1)"
```

Then open that URL in a **private window**. Your CLI credentials make everything
look fine from the terminal; a logged-out browser is what a worker is. A 403 here
is the failure mode that silently ruins a batch.

No CORS configuration is needed — plain `<img src>` is not subject to CORS. If
you find yourself editing a CORS policy you are solving the wrong problem; it is
almost always Block Public Access.

---

## 4. Build the six MTurk projects

**Create → New Project → Other → Other.** Not *Survey* or *Survey Link*: those
ship starter widgets and a completion-code field that survive into your HIT.
You can tell them apart by the field labels — *Other* says "Reward per
assignment", the survey templates say "Reward per response" (which means the
same thing: one worker completing one whole page, not one question).

Identical properties for all six:

| Field | Value |
|---|---|
| Reward per assignment | **$1.00** |
| Number of assignments per HIT | **3** |
| Time allotted per assignment | **30 minutes** |
| HIT expires in | **7 days** |
| Auto-approve and pay Workers in | **1 hour** |
| Require Masters | unchecked — costs +5% and shrinks the pool |
| Project contains adult content | unchecked |
| Qualifications | **none in the sandbox**; production: approval ≥98%, ≥1000 HITs approved, Location US |

Per project:

| Project name | Title | Description | HTML | CSV | HITs |
|---|---|---|---|---|---|
| `ko_color` | Match the color | Look at a colored square, then pick which of two pictures matches that color. 16 quick questions, about 3 minutes. | `hit_ko_color.html` | `mturk_input_ko_color.csv` | 3 |
| `ko_style` | Which one looks like a drawing? | Pick which of two pictures looks more like a drawing than a real photograph. 18 quick questions, about 4 minutes. | `hit_ko_style.html` | `mturk_input_ko_style.csv` | 3 |
| `ko_human` | Which one is the same person? | Look at a photo of a person, then pick which of two photos shows the same person. 20 quick questions, about 4 minutes. | `hit_ko_human.html` | `mturk_input_ko_human.csv` | 3 |
| `i2i2i_color` | Match the color | Look at a colored square, then pick which of two pictures matches that color. 20 quick questions, about 4 minutes. | `hit_i2i2i_color.html` | `mturk_input_i2i2i_color.csv` | 3 |
| `i2i2i_style` | Which one looks like a drawing? | Pick which of two pictures looks more like a drawing than a real photograph. 18 quick questions, about 4 minutes. | `hit_i2i2i_style.html` | `mturk_input_i2i2i_style.csv` | 2 |
| `i2i2i_human` | Which one is the same person? | Look at a photo of a person, then pick which of two photos shows the same person. 20 quick questions, about 4 minutes. | `hit_i2i2i_human.html` | `mturk_input_i2i2i_human.csv` | 3 |

Keywords for all six: `image, comparison, picture, quick, survey, research`

### `aws: command not found` on a different cluster node

The CLI is not missing, it is off `PATH`. `uv tool install` puts its launchers in
`$HOME/.local/bin` — AFS, so shared across machines — while the cluster
`.bashrc` only adds the *scratch* `.local/bin`, which is where `uv` itself
lives. Credentials in `~/.aws/` are on AFS too and follow you around. Fix:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && . ~/.bashrc
aws sts get-caller-identity        # confirms the keys are readable
```

The awscli venv sits under `/data/scratch/$USER/.local/share/uv/tools/`, which
is NFS rather than node-local, so it resolves from any node.

### Round 3: the three paired `ref→image` projects

The `ref→image` arm was rebuilt as a two-option forced choice (see the README).
A two-option layout cannot share a HIT with the four-option one, so it ships as
**three additional projects**. The six above are unchanged — do not touch them,
their round-2 results stand, and re-publishing would only re-collect data you
already have.

Same properties table as above. Only the three rows are new:

| Project name | Title | Description | HTML | CSV | HITs |
|---|---|---|---|---|---|
| `ko_color_pair` | Which image matches the target color? | A uniform target color is shown at the top. Pick which of the two images below contains an object that roughly follows that color. 16 quick questions, about 3 minutes. | `hit_ko_color_pair.html` | `mturk_input_ko_color_pair.csv` | 2 |
| `ko_style_pair` | Which image is the drawing? | Two images, one drawn and one photographed. Pick the drawing. 18 quick questions, about 4 minutes. | `hit_ko_style_pair.html` | `mturk_input_ko_style_pair.csv` | 2 |
| `ko_human_pair` | Which image shows the target person? | A target person is shown at the top. Pick which of the two images below shows that same person. 19 quick questions, about 4 minutes. | `hit_ko_human_pair.html` | `mturk_input_ko_human_pair.csv` | 2 |

6 HITs x 3 assignments x $1.00 = **$18.00 + $3.60 fee = $21.60**.

`select_subset.py` exported new images for the foils, so **re-sync S3 before
publishing** or every foil renders as a broken image:

```bash
aws s3 sync results_v4/human_eval/images "s3://$BUCKET/images" \
    --exclude '*' --include '*.jpg' --cache-control max-age=604800
```

The sync is additive — the images the live projects already reference keep their
names, because `export()` derives each name from the source path. That also
means the object count is **not** a useful check: the bucket accumulates images
from every earlier export (561 objects for 525 currently referenced). Verify
coverage instead, which is the thing that actually breaks a HIT:

```bash
aws s3 ls "s3://$BUCKET/images/" | awk '{print $4}' | sort > /tmp/in_bucket.txt
python - <<'PY'
import csv, pathlib
need = {v.rsplit('/', 1)[1]
        for p in pathlib.Path('results_v4/human_eval').glob('mturk_input_*.csv')
        for r in csv.DictReader(p.open())
        for k, v in r.items() if k.startswith('img_')}
have = {l.strip() for l in open('/tmp/in_bucket.txt') if l.strip()}
print(f"{len(need)} referenced, missing: {need - have or 'none'}")
PY
```

Titles repeat across methods because the worker-facing question genuinely is
identical; the *project name* is what keeps them apart for you.

**Setting qualifications in the sandbox will hide your own HIT from you** — a new
sandbox worker account has zero approved HITs and cannot meet "≥1000 HITs
approved". Leave them empty there.

### Pasting the layout

The *Other* template opens **Design Layout** in HTML source mode already. Select
all, delete the starter content, paste the whole `hit_<group>.html`. First line
must be the `crowd-html-elements.js` `<script>`; last must be `</crowd-form>`.

Preview at this stage shows broken images and literal `${img_top_0}` text. That
is correct — placeholders only bind to CSV data at publish time.

If you see literal `<div class="trial">` as visible text, the paste was treated
as plain text; undo and re-paste.

---

## 5. Publish, in the sandbox first

**Publish Batch** → upload `mturk_input_<group>.csv` → **Next** → *Preview HITs*
→ **Next** → *Confirm and Publish Batch*.

The upload screen only validates the file (`Line Count: 3` means a header plus
2 HITs). The cost confirmation is two screens later. If you can't find *Next*,
zoom the browser out — the requester UI clips its navigation buttons at narrow
widths.

**Check the HIT count on the confirm screen against the table above, every
time.** Publishing is always additive: it creates new HITs and never edits
existing ones, and editing a project afterwards does not change already-published
HITs. Publish the same CSV twice and you have two batches and twice the cost. In
the sandbox that's only confusion; in production it's money.

If you need to change a layout, **cancel the batch first**, then republish. Also
delete the stale project, so you don't have two same-named projects with
different layouts.

### Then do a HIT yourself

<https://workersandbox.mturk.com> → find the title → **Accept** → answer
everything → **Submit**. Check:

- every question shows 3 images, none broken
- exactly one radio stays selected — click "Both", then "Left", and confirm the
  first clears
- questions do not alternate in an obvious A-B-A-B rhythm
- submitting with a blank question is refused
- **time yourself.** If 20 questions takes much over 4 minutes, raise the reward
  before production. Workers who find a task underpaid simply don't accept it.

Do this for all six. Three separate bugs were caught this way that no amount of
structural validation found.

---

## 6. Read the results back

**Manage → Results → Download CSV.** One row per assignment.

The answers are **not** in per-question columns. `crowd-form` serialises the
entire form into a single `Answer.taskAnswers` cell holding a JSON list with one
object, and each radio group arrives as a boolean per option:

```json
[{"q0_choice": {"both": true, "left": false, "neither": false, "right": false},
  "q0_item_id": "11051495cfda",
  "q1_choice": {"both": false, "left": true, "neither": false, "right": false},
  "q1_item_id": "dc49c19a4ead", ...}]
```

`analyze_results.py` parses this. The check that matters is that
`q<k>_item_id` is present and joins to `manifest.csv` — that hidden field is the
only link from a click back to an experimental cell.

```bash
# drop every downloaded batch CSV in results_v4/human_eval/, then:
uv run python -m scripts.human_eval.analyze_results --results results_v4/human_eval/
```

Batch downloads sit directly in `human_eval/` alongside the study definition.
The analyzer globs `*.csv` and skips anything with no `Answer.taskAnswers`
column, so `manifest.csv`, the `mturk_input_*` files and the pilot are ignored
without any special-casing.

Outputs: agreement rate per condition / family / method / overall with
worker-bootstrapped CIs, worker screening on the check arms (`_check` and
`_paircheck`), position-bias and inter-rater diagnostics, plus
`human_verdicts.csv` and `disagreements.csv`, the latter listing every item
where humans and the judge differ, with image names for eyeballing. Collected
results are written up in
[`results_v4/human_eval/README.md`](../../results_v4/human_eval/README.md).

Run it on the sandbox data too. One assignment gives meaningless rates, but it
proves the parse and the join before real money moves.

---

## 7. Production

1. **COUHES sign-off first.** Human-subjects research; likely exempt, but the
   determination is theirs. The sandbox needs nothing — no real participants.
2. Rebuild all six projects on <https://requester.mturk.com>. Sandbox projects do
   not transfer. Same settings, but **add the qualifications** this time.
3. **Publish one project first** — `ko_color`, 3 HITs, ~$10.80. Let it complete,
   download, run the analyzer. If `_check` items are being failed or an arm comes
   back 100% one-sided, you've learned it cheaply.
4. Publish the other five.
5. Approve the work. Auto-approve handles it in an hour. **Reject sparingly** —
   rejections damage workers' qualification scores, and the `_check` arms let you
   exclude bad data in analysis while still paying.
6. Delete the bucket once every HIT is complete and approved:
   `aws s3 rb "s3://$BUCKET" --force`. Not before — a worker mid-assignment on a
   dead bucket sees blank boxes.

Note that auto-approval delay does **not** gate your data. Results are available
the moment a worker submits; the delay only governs automatic payment.

---

## Alternatives to MTurk

Steps 1–3 produce a platform-agnostic study. Only publish-and-collect is
MTurk-specific.

- **[Turkle](https://github.com/hltcoe/turkle)** — self-hosted, consumes the
  MTurk-format HTML template and input CSV nearly unchanged. Closest drop-in; you
  supply the participants.
- **[Prolific](https://www.prolific.com/)** — better pool for academic work,
  higher per-participant cost, layout needs porting.
- **[CloudResearch Connect](https://www.cloudresearch.com/)** — similar.

In every case `manifest.csv` stays the join key and `analyze_results.py` needs
only its input parser swapped.

---

## Gotchas, collected

| Symptom | Cause |
|---|---|
| `AccessControlListNotSupported` on upload | `--acl public-read`; use a bucket policy |
| Images 403 in a private window | Block Public Access still on, or policy missing |
| Layout shows literal `<div>` text | Pasted into WYSIWYG instead of source view |
| Multiple radios selected at once | `crowd-radio-button` siblings sharing one `name`; use native `<input type="radio">` |
| `Answer.q0_item_id` missing | Hidden inputs stripped by the WYSIWYG editor |
| No *Next* button after CSV upload | Buttons clipped; zoom out, maximise, disable ad blockers |
| Batch previews the wrong HIT count | Wrong CSV, or Excel mangled it — never round-trip these through Excel |
| Can't find your own sandbox HIT | Qualifications set in the sandbox; remove them |
| "Exceeded your monthly credit limit" | New-account limit of zero; request an increase |
| Duplicate batches after a fix | Publishing is additive; cancel the old batch and delete the stale project |
