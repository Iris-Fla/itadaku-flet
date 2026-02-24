from pathlib import Path
import argparse

from optimum.intel import OVModelForVisualCausalLM
from transformers import AutoProcessor


def normalize_lang_code(value: str) -> str:
    lang_aliases = {
        "english": "en",
        "en": "en",
        "japanese": "ja",
        "ja": "ja",
        "jp": "ja",
        "ja-jp": "ja",
    }
    key = value.strip().lower().replace("_", "-")
    return lang_aliases.get(key, key)


def build_messages(text: str, source_lang_code: str, target_lang_code: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": source_lang_code,
                    "target_lang_code": target_lang_code,
                    "text": text,
                }
            ],
        }
    ]


def translate(
    text: str,
    model_dir: Path,
    source_lang: str = "en",
    target_lang: str = "ja",
) -> str:
    if not (model_dir / "openvino_language_model.xml").exists():
        raise FileNotFoundError(f"openvino_language_model.xml not found in: {model_dir}")

    source_lang_code = normalize_lang_code(source_lang)
    target_lang_code = normalize_lang_code(target_lang)

    model = OVModelForVisualCausalLM.from_pretrained(str(model_dir), device="CPU")
    processor = AutoProcessor.from_pretrained(str(model_dir))

    messages = build_messages(text, source_lang_code, target_lang_code)
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(text=prompt, return_tensors="pt")
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
    )
    prompt_len = inputs["input_ids"].shape[1]
    return processor.batch_decode(generated_ids[:, prompt_len:], skip_special_tokens=True)[0].strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate text with local TranslateGemma OpenVINO model")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("translategemma-12b-ovt"),
        help="OpenVINO model directory",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Please restart from the bottom left of the screen to apply the settings. Once the restart is complete, the changes you made will be reflected.",
        help="Text to translate",
    )
    parser.add_argument("--source", type=str, default="en", help="Source language code (e.g. en)")
    parser.add_argument("--target", type=str, default="ja", help="Target language code (e.g. ja)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")

    translated = translate(
        text=args.text,
        model_dir=args.model_dir,
        source_lang=args.source,
        target_lang=args.target,
    )
    print(translated)


if __name__ == "__main__":
    main()