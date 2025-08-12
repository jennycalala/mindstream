# Mindstream

Generate daily art guided by your browsing history using GPT for prompts and images, with optional local generative overlays. Supports JSON/CSV history inputs, caching, and cross-platform usage.

## Features
- GPT-assisted prompt merging from visited pages
- Image generation via `gpt-image-1`
- Three modes: `abstract`, `gpt`, `hybrid`
- Styles: `abstract`, `figurative`, `realistic` (less abstract → figurative/realistic)
- Deterministic or varied results with `--seed`
- Caching for page summaries and GPT images

## Requirements
- Python 3.9+
- Dependencies in `requirements.txt`
- OpenAI API key for non-`--dry-run` usage: `OPENAI_API_KEY`

## Install
### macOS/Linux (bash/zsh)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows (PowerShell)
```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configure
- Set your API key (skip if using `--dry-run`):
  - macOS/Linux:
    ```bash
export OPENAI_API_KEY="sk-..."
    ```
  - Windows PowerShell:
    ```powershell
$env:OPENAI_API_KEY = "sk-..."
    ```

## Run
Script: `mindstream_art.py`

### macOS/Linux
```bash
python mindstream_art.py history.json \
  --date 2025-08-07 \
  --mode hybrid \
  --style realistic \
  --seed 42 \
  --out out.png \
  --log

# Or auto-read Chrome history directly (no file needed; Chrome must be closed)
python mindstream_art.py \
  --auto-history chrome \
  --date 2025-08-07 \
  --mode hybrid \
  --style realistic \
  --seed 42 \
  --out out.png \
  --log
```

### Windows PowerShell
```powershell
py .\mindstream_art.py .\history.json \
  --date 2025-08-07 \
  --mode hybrid \
  --style realistic \
  --seed 42 \
  --out .\out.png \
  --log

# Or auto-read Chrome history directly (no file needed; Chrome must be closed)
py .\mindstream_art.py \
  --auto-history chrome \
  --date 2025-08-07 \
  --mode hybrid \
  --style realistic \
  --seed 42 \
  --out .\out.png \
  --log
```

## Modes
- `abstract`: Local flow-field generative art themed by the daily prompt (no remote image).
- `gpt`: GPT-generated image only.
- `hybrid`: Combines GPT image and local overlay. For non-abstract styles, uses the GPT image directly (no overlay), which reduces abstractness.

## Key Flags
- `--style` `abstract|figurative|realistic`: Controls prompt guidance and hybrid behavior. Use `figurative`/`realistic` for less abstract outputs.
- `--seed` `<int>`: Seed for reproducibility/variation. Also tags the GPT prompt to nudge unique generations per seed.
- `--refresh-cache`: Re-summarize pages (ignore `.mindstream_cache.json`).
- `--refresh-image`: Force regenerate GPT image (ignore existing cached image file).
- `--dry-run`: Skips GPT calls; writes a stub image and prompt for quick testing.

## Caching
- Page summaries: `.mindstream_cache.json`
- GPT images: `cache/images/YYYY-MM-DD_gpt_<style>[_<seed>].png`

## Examples
- Quick test (no API calls):
  ```bash
python mindstream_art.py history.json --date 2025-08-07 --mode hybrid --style figurative --seed 1 --out out.png --dry-run --log
  ```
- Fresh, less-abstract hybrid image (realistic):
  ```bash
python mindstream_art.py history.json --date 2025-08-07 --mode hybrid --style realistic --seed 42 --refresh-image --out out.png --log
  ```
- GPT-only, realistic:
  ```bash
python mindstream_art.py history.json --date 2025-08-07 --mode gpt --style realistic --seed 7 --refresh-image --out out.png --log
  ```
- Pure abstract overlay only:
  ```bash
python mindstream_art.py history.json --date 2025-08-07 --mode abstract --seed 99 --out out.png --log
  ```

## Outputs
- Image: as specified by `--out` (e.g., `out.png`)
- Prompt used: saved next to the output file as `<out>_prompt.txt` (e.g., `out_prompt.txt`)

## Notes
- Uses OpenAI Python SDK 1.x (`from openai import OpenAI`) and `gpt-image-1` for images.
- On some macOS setups, you may see an `urllib3` OpenSSL warning; it is typically non-blocking.
