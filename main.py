import os
import base64
import asyncio
import json
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
        model="claude-sonnet-4-6",
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


@app.post("/transform")
async def transform(req: TransformRequest):
    import io

    image_data = req.image
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]
    image_bytes = base64.b64decode(image_data)

    async with httpx.AsyncClient(timeout=180.0) as client:
        # ── Build multipart: room photo is the base image, product thumbnails
        #    are reference images so the room is furnished with REAL products. ──
        files = [("image[]", ("room.jpg", io.BytesIO(image_bytes), "image/jpeg"))]

        product_lines = []
        for idx, p in enumerate(req.products):
            label = f"{p.category}: {p.name}"
            product_lines.append(f"{idx + 1}. {label}")
            if p.thumbnail:
                try:
                    timg = await client.get(p.thumbnail, timeout=15.0)
                    if timg.status_code == 200 and timg.content:
                        files.append(
                            ("image[]", (f"product_{idx}.jpg", io.BytesIO(timg.content), "image/jpeg"))
                        )
                except Exception:
                    pass  # skip thumbnails that fail to download

        # ── Compose a prompt that ties the generated room to the real products ──
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

        resp = await client.post(
            "https://api.openai.com/v1/images/edits",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            files=files,
            data={
                "model": "gpt-image-2",
                "prompt": prompt[:4000],
                "n": "1",
                "size": "1024x1024",
            },
        )

        if resp.status_code != 200:
            raise HTTPException(status_code=500, detail=f"OpenAI error: {resp.text[:300]}")

        result = resp.json()
        b64 = result["data"][0].get("b64_json")
        if b64:
            return {"imageUrl": f"data:image/png;base64,{b64}"}

        image_url = result["data"][0].get("url")
        if image_url:
            return {"imageUrl": image_url}

        raise HTTPException(status_code=500, detail="No image in OpenAI response")


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


async def search_serpapi(query: str, budget_max: int) -> dict | None:
    """Search Google Shopping India via SerpAPI and return the best result."""
    async with httpx.AsyncClient(timeout=15.0) as client:
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


@app.post("/products")
async def products(req: ProductsRequest):
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
        model="claude-sonnet-4-6",
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
        result = await search_serpapi(cat["searchQuery"], budget_max)
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

    results = await asyncio.gather(*[enrich(c) for c in categories])
    return list(results)
