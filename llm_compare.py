import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# /model 디렉토리에 있는 모델들을 리스트업합니다.
model_dir = "d:/kjy/models/"
model_names = [name for name in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, name))]

# 사용자가 원하는 모델을 다중 선택하도록 합니다.
selected_models = st.multiselect("모델을 선택해주세요.", model_names)

# 프롬프트를 입력받습니다.
token_key = st.text_input("Huggingface의 토큰 Key를 입력해주세요.")
prompt = st.text_input("프롬프트를 입력해주세요.")

if st.button("실행"):
    results = {}
    for model_name in selected_models:
        # 각 모델에 대한 토큰 Key를 입력받습니다.
        # token_key = st.text_input("Huggingface의 토큰 Key를 입력해주세요.")
        # 각 모델을 로드합니다.
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, model_name), use_auth_token=token_key)
        model = AutoModelForCausalLM.from_pretrained(os.path.join(model_dir, model_name), use_auth_token=token_key)
        # 주어진 프롬프트로 모델을 실행합니다.
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        output = model.generate(input_ids, max_length=100)
        result = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        results[model_name] = result
    # 결과를 출력합니다.
    for model_name, result in results.items():
        st.write(f"모델 {model_name}의 결과:")
        st.write(result)
