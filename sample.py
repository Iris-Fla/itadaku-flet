from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer, pipeline

model_id = "OpenVINO/Qwen2.5-7B-Instruct-int4-ov"
model = OVModelForCausalLM.from_pretrained(model_id, device="CPU")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 1. パイプラインの作成
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# 2. 翻訳対象のテキスト
text_to_translate = "The quick brown fox jumps over the lazy dog."

# 3. チャット形式（Messages）で厳格に指示する
messages = [
    {"role": "system", "content": "You are a professional translator. Output ONLY the translated text without any explanations, pronunciations, or notes."},
    {"role": "user", "content": f"Translate this to Japanese: {text_to_translate}"}
]

# プロンプトの生成
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 4. 推論実行
results = pipe(
    prompt,
    max_new_tokens=100,
    do_sample=False,        # 最も確率の高い単語を選択（決定論的）
    return_full_text=False, # 入力文（プロンプト）を隠す
)

# 5. 後処理：改行が含まれた場合に備えて、最初の一行だけを抽出
final_output = results[0]['generated_text'].strip().split('\n')[0]

print(final_output)