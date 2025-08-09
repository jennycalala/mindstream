#!/usr/bin/env python3
"""
Daily Vibe Art — GPT-powered, JSON/CSV, cached, Windows-friendly

USAGE (Windows PowerShell):
    # 1) Create & activate venv
    py -3 -m venv .venv
    .\.venv\Scripts\Activate.ps1

    # 2) Install deps
    pip install -r requirements.txt

    # 3) Set your key
    $env:OPENAI_API_KEY = "sk-..."

    # 4) Run (JSON or CSV)
    py .\daily_vibe_art.py .\history.json --date 2025-08-07 --mode hybrid --out .\out.png --log

MODES:
    abstract  - Local flow-field generative art themed by GPT daily prompt (no remote image).
    gpt       - GPT-generated abstract image only.
    hybrid    - GPT image + local flow-field overlay (palette from GPT image).

DEPENDENCIES (see requirements.txt):
    pandas, numpy, python-dateutil, scikit-learn,
    requests, trafilatura, beautifulsoup4, lxml,
    vaderSentiment, Pillow, colorthief,
    openai>=1.0.0

NOTES:
- Uses OpenAI Python SDK 1.x pattern (from openai import OpenAI; client = OpenAI()).
- gpt-image-1 returns base64; we decode and save locally.
- Caches page summaries in .daily_vibe_cache.json
- Always regenerates GPT image in hybrid mode (and saves to cache/images/YYYY-MM-DD_gpt.png)
"""

import os, sys, json, argparse, hashlib, random, base64
from io import BytesIO
from datetime import datetime, timezone
from typing import List, Tuple

import requests
import pandas as pd
import numpy as np
from dateutil import parser as dateparser
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from PIL import Image, ImageDraw, ImageOps
from colorthief import ColorThief
import trafilatura

# OpenAI 1.x client
try:
    from openai import OpenAI
except Exception:
    print("Error: OpenAI SDK not installed or too old. Run: pip install 'openai>=1.0.0'", file=sys.stderr)
    sys.exit(2)

CACHE_FILE = ".daily_vibe_cache.json"
IMAGE_CACHE_DIR = os.path.join("cache", "images")

# ---------------------- Logging ----------------------
def log(msg: str, enabled: bool):
    if enabled:
        print(msg)

# --------------------- Cache I/O ---------------------
def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

# ----------------- History loading ------------------
def load_history(path: str, log_enabled=False) -> pd.DataFrame:
    log(f"Loading history: {path}", log_enabled)
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rows = []
        for entry in data:
            title = entry.get("title", "")
            url = entry.get("url", "")
            ts = entry.get("visitTime") or entry.get("lastVisitTime")
            if not url or not ts:
                continue
            try:
                # timezone-aware UTC (avoid utcfromtimestamp deprecation)
                dt = datetime.fromtimestamp(float(ts) / 1000.0, tz=timezone.utc)
            except Exception:
                continue
            rows.append({"title": title, "url": url, "datetime": dt})
        df = pd.DataFrame(rows)
    elif path.lower().endswith(".csv"):
        df = pd.read_csv(path)
        # Try to normalize datetime column
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=False)
        elif "timestamp" in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=False)
        else:
            raise ValueError("CSV must include 'datetime' or 'timestamp' column.")
        if "url" not in df.columns:
            raise ValueError("CSV must include 'url' column.")
        if "title" not in df.columns:
            df["title"] = ""
    else:
        raise ValueError("Unsupported file format. Use .json or .csv")

    # Ensure datetimelike dtype for .dt accessor
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=False)
    df = df.dropna(subset=["datetime"])
    return df

# --------------- Web fetch & clean ------------------
def fetch_page_text(url: str, log_enabled=False) -> str:
    try:
        log(f"Fetching: {url}", log_enabled)
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        text = trafilatura.extract(resp.text) or ""
        return text.strip()
    except Exception as e:
        log(f"  fetch failed: {e}", log_enabled)
        return ""

# --------------- GPT helpers (text) -----------------
def gpt_summarize_page(client: OpenAI, text: str, title: str, url: str, log_enabled=False) -> str:
    log(f"Summarizing via GPT: {title or url}", log_enabled)
    system_prompt = (
        "You are an abstract art inspiration assistant. "
        "From this page’s text, describe non-literal shapes, colors, and moods for abstract art. "
        "Avoid copying identifiable characters, logos, or literal brand imagery. "
        "Return 1–2 sentences."
    )
    user_prompt = f"Title: {title}\nURL: {url}\nContent:\n{text[:8000]}\n\nAbstract art inspiration:"
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()

def gpt_merge_daily_prompt(client: OpenAI, page_summaries: List[str], log_enabled=False) -> str:
    log("Merging daily concept via GPT…", log_enabled)
    system_prompt = (
        "You are an abstract art concept creator. "
        "From the following abstract inspirations, create ONE cohesive abstract art prompt "
        "that captures the dominant mood, shapes, and colors for a single daily artwork. "
        "Keep it non-literal and avoid brand characters/logos."
    )
    joined = "\n".join(f"- {s}" for s in page_summaries)
    user_prompt = f"Abstract inspirations:\n{joined}\n\nFinal daily abstract art prompt:"
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.8,
    )
    return resp.choices[0].message.content.strip()

# -------------- GPT image generation ----------------
def gpt_generate_image(client: OpenAI, prompt: str, out_path: str, log_enabled=False):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    log("Generating abstract image via gpt-image-1…", log_enabled)

    # Determine best size based on desired output dimensions (default to wide)
    try:
        # If file already exists and is not empty, skip
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            log(f"Using cached image → {out_path}", log_enabled)
            return out_path
    except Exception:
        pass

    # Guess output aspect ratio from filename pattern or default
    width, height = 1536, 1024  # default wide
    if out_path.lower().endswith(".png"):
        # Optional: in the future, parse --out size from arguments
        pass

    # Map to nearest allowed size
    allowed_sizes = [
        (1024, 1024),
        (1024, 1536),
        (1536, 1024),
    ]
    aspect = width / height
    chosen_size = min(allowed_sizes, key=lambda s: abs((s[0] / s[1]) - aspect))
    size_str = f"{chosen_size[0]}x{chosen_size[1]}"

    log(f"Using size: {size_str}", log_enabled)

    result = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size=size_str,
    )
    b64 = result.data[0].b64_json
    img_bytes = base64.b64decode(b64)
    with open(out_path, "wb") as f:
        f.write(img_bytes)
    return out_path

# --------------- Palette extraction -----------------
def extract_palette(image_path: str, num_colors: int = 5) -> List[Tuple[int,int,int]]:
    with open(image_path, "rb") as f:
        ct = ColorThief(f)
        return ct.get_palette(color_count=num_colors)

# -------- Flow-field (simple, Windows-safe) ---------
def generate_flow_field_image(size, palette, out_path, overlay_path=None, overlay_opacity=128):
    w, h = size
    base = Image.new("RGB", (w, h), (250, 250, 250))
    draw = ImageDraw.Draw(base)

    # simple seeded randomness for determinism
    rnd = random.Random(42)
    for _ in range(3000):
        x, y = rnd.randint(0, w), rnd.randint(0, h)
        dx, dy = rnd.randint(-12, 12), rnd.randint(-12, 12)
        color = palette[rnd.randrange(len(palette))]
        draw.line((x, y, x + dx, y + dy), fill=tuple(color), width=1)

    if overlay_path and os.path.exists(overlay_path):
        fg = Image.open(overlay_path).convert("RGBA").resize((w, h))
        # apply uniform alpha safely on Windows
        alpha = Image.new("L", fg.size, color=int(overlay_opacity))
        fg.putalpha(alpha)
        base = Image.alpha_composite(base.convert("RGBA"), fg).convert("RGB")

    base.save(out_path, "PNG")

# ------------------------- Main ---------------------
def main():
    ap = argparse.ArgumentParser(description="Generate daily abstract art from browser history")
    ap.add_argument("history", help="Path to history JSON or CSV")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--mode", choices=["abstract", "gpt", "hybrid"], default="hybrid")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--refresh-cache", action="store_true", help="Re-summarize all pages (ignore cache)")
    ap.add_argument("--log", action="store_true", help="Print progress logs")
    ap.add_argument("--dry-run", action="store_true", help="Skip GPT calls; stub summaries for testing")
    args = ap.parse_args()

    # OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.stderr.write("Error: OPENAI_API_KEY not set.\n")
        sys.exit(2)
    client = OpenAI(api_key=api_key)

    # Load data
    df = load_history(args.history, log_enabled=args.log)
    target_date = dateparser.parse(args.date).date()

    # Ensure datetimelike and filter by date
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=False)
    df = df.dropna(subset=["datetime"])
    day_mask = df["datetime"].dt.date == target_date
    df_day = df[day_mask].copy()
    if df_day.empty:
        sys.stderr.write(f"No entries found for date {target_date}.\n")
        sys.exit(1)

    cache = load_cache()
    page_summaries: List[str] = []

    for _, row in df_day.iterrows():
        url = str(row.get("url", ""))
        title = str(row.get("title", ""))
        key = f"{target_date}|{url}"

        if args.dry_run:
            summary = f"Abstract mood from: {title or url}"
        elif (not args.refresh_cache) and key in cache:
            summary = cache[key]
            log(f"[cache] {title or url}", args.log)
        else:
            text = fetch_page_text(url, log_enabled=args.log)
            summary = gpt_summarize_page(client, text, title, url, log_enabled=args.log)
            cache[key] = summary

        page_summaries.append(summary)

    if not args.dry_run:
        save_cache(cache)

    # Merge into daily prompt
    daily_prompt = (
        " | ".join(page_summaries)
        if args.dry_run
        else gpt_merge_daily_prompt(client, page_summaries, log_enabled=args.log)
    )

    # Save prompt next to output
    prompt_txt_path = os.path.splitext(args.out)[0] + "_prompt.txt"
    with open(prompt_txt_path, "w", encoding="utf-8") as f:
        f.write(daily_prompt)

    # Ensure image cache dir
    os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)
    gpt_img_cache = os.path.join(IMAGE_CACHE_DIR, f"{target_date}_gpt.png")

    if args.mode == "gpt":
        if args.dry_run:
            # stub image
            generate_flow_field_image((1920, 1080), [(200, 200, 200)] * 5, args.out)
        else:
            gpt_generate_image(client, daily_prompt, args.out, log_enabled=args.log)

    elif args.mode == "abstract":
        # Deterministic palette from prompt hash
        seed = int(hashlib.sha1(daily_prompt.encode("utf-8")).hexdigest()[:8], 16)
        rnd = random.Random(seed)
        palette = [(rnd.randrange(40, 220), rnd.randrange(40, 220), rnd.randrange(40, 220)) for _ in range(5)]
        generate_flow_field_image((1920, 1080), palette, args.out)

    elif args.mode == "hybrid":
        if args.dry_run:
            # stub palette + overlay-less render
            palette = [(60,120,200), (200,80,120), (120,200,100), (80,80,180), (180,160,90)]
            generate_flow_field_image((1920, 1080), palette, args.out)
        else:
            # Always regenerate GPT image, then extract palette and overlay generative lines
            gpt_generate_image(client, daily_prompt, gpt_img_cache, log_enabled=args.log)
            palette = extract_palette(gpt_img_cache, num_colors=6)
            generate_flow_field_image((1920, 1080), palette, args.out, overlay_path=gpt_img_cache, overlay_opacity=128)

    log(f"Saved image → {args.out}", args.log)
    log(f"Saved prompt → {prompt_txt_path}", args.log)

if __name__ == "__main__":
    main()
