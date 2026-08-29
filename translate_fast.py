import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def main():
    parser = argparse.ArgumentParser(description="Fast Interactive Translator (English to Bemba/Chewa)")
    parser.add_argument("--lang", type=str, choices=["bem", "nya"], required=True, 
                        help="Target language: 'bem' for Bemba, 'nya' for Nyanja/Chewa")
    args = parser.parse_args()
    
    target_lang_code = "bem_Latn" if args.lang == "bem" else "nya_Latn"
    
    print(f"Loading model into memory... (This takes a few seconds, but only happens ONCE)")
    
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    print("\n✅ Model loaded! Ready for fast translation.")
    print("Type your English sentence and press Enter. Type 'quit' to exit.\n")
    
    while True:
        try:
            text = input("English: ")
            if text.strip().lower() in ['quit', 'exit', 'q']:
                break
            if not text.strip():
                continue
                
            inputs = tokenizer(text, return_tensors="pt")
            
            translated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang_code),
                max_length=400
            )
            
            translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            print(f"{args.lang.upper():>7}: {translation}\n")
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
