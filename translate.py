import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def main():
    parser = argparse.ArgumentParser(description="Translate English to Bemba or Chewa/Nyanja")
    parser.add_argument("text", type=str, help="The English text to translate")
    parser.add_argument("--lang", type=str, choices=["bem", "nya"], required=True, 
                        help="Target language: 'bem' for Bemba, 'nya' for Nyanja/Chewa")
    
    args = parser.parse_args()
    
    # NLLB language codes
    # Bemba: bem_Latn
    # Nyanja/Chewa: nya_Latn
    target_lang_code = "bem_Latn" if args.lang == "bem" else "nya_Latn"
    
    print(f"Loading translation model (this might take a moment the first time to download)...")
    
    # We use facebook's NLLB (No Language Left Behind)
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    inputs = tokenizer(args.text, return_tensors="pt")
    
    # Translate to the target language
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang_code),
        max_length=400
    )
    
    translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    
    print("\n--- Translation ---")
    print(f"English : {args.text}")
    print(f"{args.lang.upper()}     : {translation}")
    print("-------------------\n")
    
    print("Note: To improve accuracy on specific local dialects, you can fine-tune this model using the Liseli dataset as outlined in the previous guide.")

if __name__ == "__main__":
    main()
