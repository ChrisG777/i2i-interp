Project site for the paper **"Vision-Language Binding in In-Context Image Generation"**.

Template forked from the [Nerfies project page](https://github.com/nerfies/nerfies.github.io). Licensed under [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/).

## Updating the animations

PowerPoint exports are large (often 10x what the web needs). Workflow:

1. Drop the new uncompressed `.mp4` into `sources/` (gitignored), keeping the same filename as the file in `static/` you want to replace.
2. Run `scripts/compress-video.sh` (no args). It reads every `sources/*.mp4` and writes a web-friendly version to `static/<same-name>.mp4` (H.264, 30 fps, CRF 30, faststart). Override with `CRF=`, `FPS=`, `PRESET=` env vars if needed.
3. For Twitter/Tweetfully uploads: **manually re-encode** `sources/<name>.mp4` into `twitter/<name>.mp4` using HandBrake / QuickTime / your encoder of choice. Every automated configuration tried (silent AAC, mp42 brand, BT.709 VUI tags, x264 colorprim) still hit "greyed out / not selectable" rejections in Twitter's upload UI, so we do this by hand.
4. `git add static/*.mp4` and commit. `sources/` and `twitter/` stay local.
