from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import uvicorn
from contextlib import asynccontextmanager

# Global variables to hold the model and tokenizer in memory
tokenizer = None
model = None

# We use the lifespan to load the model exactly once when the server starts up
@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model
    print("Loading AI model into memory (this will take a few seconds)...")
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print("✅ Model loaded! API is ready to accept requests.")
    yield
    print("Shutting down API...")

app = FastAPI(lifespan=lifespan)

class TranslationRequest(BaseModel):
    text: str
    lang: str  # "bem" or "nya"

@app.post("/translate")
async def translate_text(request: TranslationRequest):
    # Map simple lang codes to NLLB codes
    target_lang_code = "bem_Latn" if request.lang == "bem" else "nya_Latn"
    
    # Tokenize and translate
    inputs = tokenizer(request.text, return_tensors="pt")
    
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang_code),
        max_length=400
    )
    
    translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    
    return {"original": request.text, "translation": translation, "lang": request.lang}

if __name__ == "__main__":
    # Start the server on port 8000
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
