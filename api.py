from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Union, List, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import uvicorn
from contextlib import asynccontextmanager
import asyncio

# Global variables to hold the model and tokenizer in memory
tokenizer = None
model = None
translation_lock = asyncio.Lock()

LANGUAGE_CODES = {
    "bem": "bem_Latn",
    "bemba": "bem_Latn",
    "bem_Latn": "bem_Latn",
    "nya": "nya_Latn",
    "nyanja": "nya_Latn",
    "chewa": "nya_Latn",
    "nya_Latn": "nya_Latn",
    "eng": "eng_Latn",
    "english": "eng_Latn",
    "eng_Latn": "eng_Latn",
}

# We use the lifespan to load the model exactly once when the server starts up
@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model
    print("Loading AI model into memory (this will take a few seconds)...")
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print("✅ Model loaded! API is ready to accept requests.")
    yield
    print("Shutting down API...")

app = FastAPI(lifespan=lifespan)

class TranslationRequest(BaseModel):
    text: Union[str, List[str]]
    lang: str = "eng"  # Target language (e.g., "eng", "bem", "nya")
    src_lang: Optional[str] = None  # Source language (e.g., "bem", "eng", "nya")

class TranslationItem(BaseModel):
    original: str
    translation: str
    source_lang: str
    target_lang: str

@app.post("/translate", response_model=List[TranslationItem])
async def translate_text(request: TranslationRequest):
    # Normalize input into a list of strings
    if isinstance(request.text, str):
        texts = [request.text]
    else:
        texts = request.text

    if not texts:
        return []

    # Target language resolution
    target_code = LANGUAGE_CODES.get(request.lang.lower())
    if not target_code:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported target language '{request.lang}'. Supported: bem, nya, eng",
        )

    # Source language resolution
    # If not explicitly provided:
    # - If target is English, default source to Bemba
    # - Otherwise, default source to English
    if request.src_lang:
        source_code = LANGUAGE_CODES.get(request.src_lang.lower())
        if not source_code:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported source language '{request.src_lang}'. Supported: bem, nya, eng",
            )
        resolved_src = request.src_lang
    else:
        if request.lang.lower() in ("eng", "english", "eng_Latn"):
            source_code = "bem_Latn"
            resolved_src = "bem"
        else:
            source_code = "eng_Latn"
            resolved_src = "eng"

    async with translation_lock:
        tokenizer.src_lang = source_code

        # Batch in chunks of 32 to maintain speed and manage memory
        BATCH_SIZE = 32
        all_translations = []

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)

            translated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_code),
                max_length=400,
            )

            batch_translations = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
            all_translations.extend(batch_translations)

    return [
        TranslationItem(
            original=orig,
            translation=trans,
            source_lang=resolved_src,
            target_lang=request.lang,
        )
        for orig, trans in zip(texts, all_translations)
    ]

if __name__ == "__main__":
    # Start the server on port 8000
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
