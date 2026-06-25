import os
import base64
import asyncio
import json
import time
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import anthropic
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Indian Minimalist Space Transformer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODELSLAB_API_KEY = os.getenv("MODELSLAB_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# Cheap structured tasks (room analysis, query generation) run on Haiku —
# ~3-5x cheaper than Sonnet with no visible quality drop for this work.
TEXT_MODEL = "claude-haiku-4-5"

SERP_CACHE_TTL = 60 * 60 * 24  # 24 hours

# ── Cache layer ──────────────────────────────────────────────────────────
# Redis-backed cache (shared across server instances, survives restarts) with
# a transparent in-memory fallback so the app keeps working if Redis is down.
# Values are JSON-serialised; keys are namespaced strings.
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

try:
    import redis.asyncio as aioredis

    _redis = aioredis.from_url(REDIS_URL, decode_responses=True)
except Exception:
    _redis = None

# Fallback store used when Redis is unreachable: {key: (expiry_ts, value)}.
_mem_cache: dict[str, tuple[float, object]] = {}


async def cache_get(key: str):
    """Return the cached value for key, or None. Tries Redis, falls back to
    the in-memory dict if Redis errors. A sentinel-free miss returns None."""
    if _redis is not None:
        try:
            raw = await _redis.get(key)
            return json.loads(raw) if raw is not None else None
        except Exception:
            pass  # fall through to in-memory
    entry = _mem_cache.get(key)
    if entry is not None:
        expiry, value = entry
        if time.time() < expiry:
            return value
        del _mem_cache[key]
    return None


async def cache_set(key: str, value, ttl: int = SERP_CACHE_TTL):
    """Store value under key with a TTL (seconds). Writes to Redis when
    available and always to the in-memory fallback so a later Redis outage
    still has something to serve."""
    if _redis is not None:
        try:
            await _redis.set(key, json.dumps(value), ex=ttl)
        except Exception:
            pass
    _mem_cache[key] = (time.time() + ttl, value)

STYLE_DETAILS = {
    "Japandi": "natural wood tones, muted sage and beige, minimal clutter, paper lamp, low furniture, zen atmosphere",
    "Indian Minimalist": "warm white walls, terracotta and brass accents, handloom textiles, clay pots, cane furniture, block print cushions",
    "Bohemian": "layered kilim rugs, macrame wall art, lush plants, rattan furniture, string lights, jewel tones",
    "Scandinavian": "white walls, light oak wood, clean lines, cozy wool throws, neutral palette, hygge",
    "Coastal": "ocean blues and whites, rattan furniture, driftwood accents, linen curtains, airy and light",
    "Industrial": "exposed brick or concrete, black metal fixtures, dark walnut, Edison bulbs, leather accents",
    "Art Deco": "velvet in emerald or navy, gold geometric accents, marble surfaces, glamorous symmetry, rich opulence",
    "Wabi-Sabi": "aged natural materials, linen, jute, weathered wood, imperfect clay pottery, earthy tones",
}


class AnalyseRequest(BaseModel):
    image: str
    style: str
    budget: str


class ProductForImage(BaseModel):
    category: str
    name: str
    thumbnail: str = ""


class TransformRequest(BaseModel):
    image: str
    sdPrompt: str
    negativePrompt: str = ""
    style: str = ""
    products: list[ProductForImage] = []


class ProductsRequest(BaseModel):
    roomType: str
    style: str
    budget: str
    keyCategories: list[str]


class RefineRequest(BaseModel):
    image: str  # current generated image (data URL) to edit
    message: str  # user's natural-language change request
    style: str = ""
    products: list[ProductForImage] = []
    history: list[dict] = []  # [{role, content}] prior chat turns


@app.get("/health")
async def health():
    """Report which cache backend is live (redis vs in-memory fallback)."""
    backend = "memory"
    if _redis is not None:
        try:
            await _redis.ping()
            backend = "redis"
        except Exception:
            backend = "memory (redis unreachable)"
    return {"status": "ok", "cache": backend}


@app.post("/analyse")
async def analyse(req: AnalyseRequest):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    style_desc = STYLE_DETAILS.get(req.style, req.style)

    image_data = req.image
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    prompt = f"""You are an expert interior designer. Analyse this room photo and return a JSON object (no markdown, pure JSON) with these exact keys:

{{
  "roomType": "e.g. Living Room, Bedroom, Kitchen, etc.",
  "currentStyle": "brief description of current style",
  "elementsDetected": ["list", "of", "detected", "furniture", "and", "decor", "items"],
  "sdPrompt": "A detailed Stable Diffusion img2img prompt to transform this room into {req.style} style. Include: {style_desc}. Be very specific about colors, materials, lighting, furniture placement. End with: interior design photography, 4k, architectural digest, professional lighting, photorealistic, high quality",
  "negativePrompt": "ugly, blurry, low quality, distorted, deformed, watermark, text, people, person, human",
  "designNarrative": "2-3 sentence poetic description of the transformed space in {req.style} style, as a designer would present it to a client",
  "keyCategories": ["5-7 product categories needed for this room in {req.style} style, e.g. Sofa, Coffee Table, Floor Lamp, etc."]
}}

Budget context: {req.budget}
Target style: {req.style}

Return ONLY the JSON object, no other text."""

    message = client.messages.create(
        model=TEXT_MODEL,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_data,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )

    try:
        result = json.loads(message.content[0].text)
        return result
    except json.JSONDecodeError:
        text = message.content[0].text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
        raise HTTPException(status_code=500, detail="Failed to parse AI response")


def _decode_image(data_url: str) -> bytes:
    """Decode a base64 data URL (or raw base64) to bytes."""
    data = data_url.split(",", 1)[1] if "," in data_url else data_url
    return base64.b64decode(data)


async def _edit_image(
    base_image_bytes: bytes,
    prompt: str,
    ref_products: list[ProductForImage],
    client: httpx.AsyncClient,
    quality: str = "",
) -> str:
    """Call gpt-image-2 to edit base_image_bytes, attaching any product
    thumbnails as reference images. Returns a data-URL string.

    Thumbnails are fetched concurrently (not in a blocking loop) with a short
    timeout so one slow image can't stall the whole request. `quality` trades
    speed/cost against fidelity — gpt-image-2 timings: low ~20s, medium ~100s,
    high ~135s. Empty string = omit (server default, ~21s). Use "low" for fast
    interactive edits; leave default for the initial hero render."""
    import io

    async def fetch_thumb(idx: int, p: ProductForImage):
        if not p.thumbnail:
            return None
        try:
            timg = await client.get(p.thumbnail, timeout=8.0)
            if timg.status_code == 200 and timg.content:
                return (idx, timg.content)
        except Exception:
            pass
        return None

    files = [("image[]", ("room.jpg", io.BytesIO(base_image_bytes), "image/jpeg"))]
    fetched = await asyncio.gather(*[fetch_thumb(i, p) for i, p in enumerate(ref_products)])
    for item in fetched:
        if item is not None:
            idx, content = item
            files.append(("image[]", (f"product_{idx}.jpg", io.BytesIO(content), "image/jpeg")))

    resp = await client.post(
        "https://api.openai.com/v1/images/edits",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        files=files,
        data={
            "model": "gpt-image-2",
            "prompt": prompt[:4000],
            "n": "1",
            "size": "1024x1024",
            **({"quality": quality} if quality else {}),
        },
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {resp.text[:300]}")

    result = resp.json()
    b64 = result["data"][0].get("b64_json")
    if b64:
        return f"data:image/png;base64,{b64}"
    image_url = result["data"][0].get("url")
    if image_url:
        return image_url
    raise HTTPException(status_code=500, detail="No image in OpenAI response")


@app.post("/transform")
async def transform(req: TransformRequest):
    image_bytes = _decode_image(req.image)

    async with httpx.AsyncClient(timeout=180.0) as client:
        product_lines = [f"{i + 1}. {p.category}: {p.name}" for i, p in enumerate(req.products)]
        style_desc = STYLE_DETAILS.get(req.style, req.style)
        if product_lines:
            prompt = (
                f"The FIRST image is a room photo. The following images are REAL furniture and decor "
                f"products that are available to buy. Redesign the room in {req.style} style "
                f"({style_desc}), furnishing it using these specific real products:\n"
                + "\n".join(product_lines)
                + "\n\nPlace these exact products naturally into the room, matching their shape, colour, "
                "material and design as shown in the reference images. Keep the room's architecture "
                "(walls, windows, floor layout) the same. Make it a cohesive, photorealistic interior. "
                "interior design photography, 4k, architectural digest, professional lighting, "
                "photorealistic, high quality"
            )
        else:
            prompt = req.sdPrompt

        image_url = await _edit_image(image_bytes, prompt, req.products, client)
        return {"imageUrl": image_url}


@app.post("/refine")
async def refine(req: RefineRequest):
    """Conversational image editor. The user describes a change in natural
    language ("swap the sofa for #2", "make the rug darker", "warmer lighting")
    and we regenerate the current image accordingly, pulling in the relevant
    real-product reference image when a swap is requested."""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    client_anthropic = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # ── Step 1: Claude turns the casual request into a precise edit instruction
    #    and picks which available products (if any) to use as references. ──
    catalog = "\n".join(
        f"{i + 1}. {p.category}: {p.name}" for i, p in enumerate(req.products)
    ) or "(no product list available)"
    history_txt = "\n".join(
        f"{h.get('role', 'user')}: {h.get('content', '')}" for h in req.history[-6:]
    )

    planner_prompt = f"""You are an interior-design image-edit assistant. The user is refining a generated {req.style} room image.

Available real products the user can place (referenced by number):
{catalog}

Recent conversation:
{history_txt or '(none)'}

User's new request: "{req.message}"

Decide how to edit the image. Return ONLY a JSON object:
{{
  "editInstruction": "a precise, visual instruction for an image editor describing exactly what to change, e.g. 'Replace the existing sofa with a low-profile cane sofa in natural wood; keep everything else identical'. Always say to keep the room architecture (walls, windows, floor) unchanged.",
  "productRefs": [list of product NUMBERS from the catalog above to use as visual references, or empty list if the change is not about a specific catalog product],
  "reply": "one short friendly sentence to show the user, e.g. 'Swapped in the cane sofa — take a look.'"
}}"""

    msg = client_anthropic.messages.create(
        model=TEXT_MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": planner_prompt}],
    )
    text = msg.content[0].text.strip()
    start, end = text.find("{"), text.rfind("}") + 1
    try:
        plan = json.loads(text[start:end])
    except Exception:
        plan = {"editInstruction": req.message, "productRefs": [], "reply": "Updated the design."}

    edit_instruction = plan.get("editInstruction", req.message)
    reply = plan.get("reply", "Updated the design.")
    refs_idx = plan.get("productRefs", []) or []

    # Map 1-based product numbers to the actual products for reference images.
    ref_products: list[ProductForImage] = []
    for n in refs_idx:
        try:
            i = int(n) - 1
            if 0 <= i < len(req.products):
                ref_products.append(req.products[i])
        except (ValueError, TypeError):
            pass

    # ── Step 2: regenerate the image from the CURRENT image + instruction ──
    base_bytes = _decode_image(req.image)
    prompt = (
        f"This is a {req.style} interior room. {edit_instruction} "
        "Keep the room's architecture (walls, windows, floor layout) and all "
        "unmentioned furniture exactly the same. Photorealistic interior design "
        "photography, 4k, architectural digest, professional lighting, high quality."
    )
    if ref_products:
        prompt += (
            " The additional images are the real product(s) to use — match their "
            "shape, colour, material and design closely."
        )

    async with httpx.AsyncClient(timeout=180.0) as client:
        # "low" keeps interactive edits fast (~20s vs ~100s for medium).
        image_url = await _edit_image(base_bytes, prompt, ref_products, client, quality="low")

    return {"imageUrl": image_url, "reply": reply}


def retailer_url(source: str, title: str, query: str) -> str:
    """Build a direct search URL on known Indian retailers, else Google Shopping."""
    s = source.lower()
    q = title or query
    if "amazon" in s:
        return f"https://www.amazon.in/s?k={httpx.URL('', params={'k': q}).params}"
    if "flipkart" in s:
        return f"https://www.flipkart.com/search?q={q.replace(' ', '+')}"
    if "pepperfry" in s:
        return f"https://www.pepperfry.com/site/search?q={q.replace(' ', '+')}"
    if "ikea" in s:
        return f"https://www.ikea.com/in/en/search/?q={q.replace(' ', '+')}"
    if "urbanladder" in s or "urban ladder" in s:
        return f"https://www.urbanladder.com/search?q={q.replace(' ', '+')}"
    # Fallback: Google Shopping product page for this query
    return f"https://www.google.com/search?tbm=shop&q={q.replace(' ', '+')}&gl=in"


async def search_serpapi(
    query: str,
    budget_max: int,
    category: str = "",
    style: str = "",
) -> dict | None:
    """Search Google Shopping India via SerpAPI and return the best result.
    Returns None on any failure (timeout, error, no results) so the caller
    can fall back gracefully without crashing the whole batch.

    Results are cached for SERP_CACHE_TTL. The cache is keyed on
    (category, style, budget) rather than the raw query string, because the
    query wording is regenerated each time and varies — keying on the stable
    (category, style, budget) triple lets the same lookup ("Japandi sofa under
    ₹50k") be reused across users even when the exact phrasing differs.
    Falls back to the query text when category/style aren't supplied."""
    if category and style:
        cache_key = f"serp:{category.strip().lower()}:{style.strip().lower()}:{budget_max}"
    else:
        cache_key = f"serp:q:{query.strip().lower()}:{budget_max}"

    cached = await cache_get(cache_key)
    if cached is not None:
        return cached

    result = await _search_serpapi_uncached(query, budget_max)
    if result is not None:
        await cache_set(cache_key, result)
    return result


async def _search_serpapi_uncached(query: str, budget_max: int) -> dict | None:
    """Actual SerpAPI call, no caching. See search_serpapi for behaviour."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                "https://serpapi.com/search",
                params={
                    "engine": "google_shopping",
                    "q": query,
                    "gl": "in",
                    "hl": "en",
                    "currency": "INR",
                    "api_key": SERPAPI_KEY,
                },
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            results = data.get("shopping_results", [])
            if not results:
                return None

            def parse_price(p: str) -> int:
                try:
                    return int("".join(c for c in p if c.isdigit())[:7])
                except Exception:
                    return 0

            affordable = [r for r in results if budget_max == 0 or parse_price(r.get("price", "0")) <= budget_max]
            candidates = affordable if affordable else results
            best = candidates[0]

            source = best.get("source", "")
            title = best.get("title", "")
            buy_link = retailer_url(source, title, query)

            return {
                "title": title,
                "price": best.get("price", ""),
                "thumbnail": best.get("thumbnail", ""),
                "link": buy_link,
                "source": source,
            }
    except Exception:
        return None


@app.post("/products")
async def products(req: ProductsRequest):
    # Whole-response cache: same room+style+budget+categories → reuse, skipping
    # both the Claude query-gen call and every SerpAPI search.
    cats = ",".join(c.strip().lower() for c in req.keyCategories[:5])
    cache_key = (
        f"products:{req.roomType.strip().lower()}:{req.style.strip().lower()}:"
        f"{req.budget.strip().lower()}:{cats}"
    )
    cached = await cache_get(cache_key)
    if cached is not None:
        return cached

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    style_desc = STYLE_DETAILS.get(req.style, req.style)

    budget_map = {
        "Under ₹10k": ("under 10000 INR total, items under 3000 INR each", 3000),
        "₹25k": ("25000 INR total, items 2000-8000 INR each", 8000),
        "₹50k": ("50000 INR total, items 3000-15000 INR each", 15000),
        "₹1 Lakh+": ("1 lakh+ INR total, premium items 5000-40000 INR each", 40000),
    }
    budget_desc, budget_max = budget_map.get(req.budget, ("mid-range Indian market", 10000))

    # Step 1: Claude generates smart search queries per category
    prompt = f"""You are an expert Indian interior designer. For a {req.roomType} in {req.style} style, generate 5 product search queries for Google Shopping India.

Style: {style_desc}
Budget: {budget_desc}
Categories needed: {", ".join(req.keyCategories[:5])}

Return a JSON array (no markdown, pure JSON) of exactly 5 objects:
{{
  "category": "e.g. Sofa, Floor Lamp, Rug",
  "emoji": "single relevant emoji",
  "description": "1 sentence on why this fits the {req.style} style",
  "searchQuery": "specific Google Shopping India search query, include style keywords and material e.g. 'cane rattan sofa natural wood japandi living room'"
}}

Make queries specific enough to find real Indian products. Return ONLY the JSON array."""

    message = client.messages.create(
        model=TEXT_MODEL,
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )

    text = message.content[0].text.strip()
    start = text.find("[")
    end = text.rfind("]") + 1
    if start == -1 or end <= start:
        raise HTTPException(status_code=500, detail="Failed to parse Claude response")

    categories = json.loads(text[start:end])

    # Step 2: Search SerpAPI for each category in parallel
    async def enrich(cat: dict) -> dict:
        result = await search_serpapi(
            cat["searchQuery"], budget_max, category=cat["category"], style=req.style
        )
        if result and result["title"]:
            return {
                "category": cat["category"],
                "emoji": cat["emoji"],
                "description": cat["description"],
                "name": result["title"],
                "price": result["price"] or "See link",
                "thumbnail": result["thumbnail"],
                "link": result["link"],
                "source": result["source"],
                "searchQuery": cat["searchQuery"],
            }
        # Fallback: no real result found, return category-only card
        return {
            "category": cat["category"],
            "emoji": cat["emoji"],
            "description": cat["description"],
            "name": f"{req.style} {cat['category']}",
            "price": "",
            "thumbnail": "",
            "link": f"https://www.google.com/search?tbm=shop&q={cat['searchQuery'].replace(' ', '+')}",
            "source": "Google Shopping",
            "searchQuery": cat["searchQuery"],
        }

    results = list(await asyncio.gather(*[enrich(c) for c in categories]))
    await cache_set(cache_key, results)
    return results
