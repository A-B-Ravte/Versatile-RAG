from llm.llamma_cpp_wrapper import LlammaLocal


model_path = r"models/LlammaLocalModel/qwen2.5-coder-7b-instruct.Q4_K_M.gguf"
llm = LlammaLocal.get_instance(model_path)

final_prompt = "write a python code to get substring from string."

response = llm(
        final_prompt,
        max_tokens=200,
        temperature=0.01,
        top_p=0.9,
        repeat_penalty=1.1
    )

print(response['choices'][0]['text'])