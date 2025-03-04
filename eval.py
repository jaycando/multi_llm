import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score as bert_score

def load_data(json_file):
    """JSON 파일에서 데이터를 로드합니다."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def generate_simple_queries(complex_query, tokenizer, model, device):
    """
    모델에게 복합질의를 두 개의 단문질의로 분리하도록 프롬프트를 구성하고,
    생성된 결과에서 단문질의를 추출합니다.
    """
    # 프롬프트 구성 (원하는 출력을 유도)
    prompt = f"복합질의를 2개의 단문질의로 분리해줘:\n{complex_query}\n단문질의 1:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # 생성: 최대 새 토큰 수, 샘플링 방법 등은 상황에 맞게 조절
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # '단문질의 2:'가 포함되어 있다면 이를 기준으로 분리합니다.
    if "단문질의 2:" in output_text:
        parts = output_text.split("단문질의 2:")
        simple1 = parts[0].split("단문질의 1:")[-1].strip()
        simple2 = parts[1].strip()
    else:
        # 없을 경우 개행 기준으로 첫 두 줄을 사용 (필요에 따라 수정)
        lines = [line.strip() for line in output_text.split('\n') if line.strip() != ""]
        if len(lines) >= 2:
            simple1, simple2 = lines[0], lines[1]
        else:
            simple1 = lines[0] if lines else ""
            simple2 = ""
    
    return simple1, simple2

def evaluate_predictions(predictions, ground_truths):
    """
    예측 결과와 정답(단문질의 쌍)에 대해 BLEU와 BERT Score를 계산합니다.
    각 예제에서 두 단문질의에 대해 각각 BLEU 점수를 산출하고 평균하며,
    전체 문장에 대해 BERT Score(F1)를 계산합니다.
    """
    bleu_scores = []
    preds_all = []
    refs_all = []
    
    for (pred1, pred2), (gt1, gt2) in zip(predictions, ground_truths):
        # 토큰화하여 BLEU 점수 계산 (각 단문질의별)
        pred1_tokens = pred1.split()
        pred2_tokens = pred2.split()
        gt1_tokens = gt1.split()
        gt2_tokens = gt2.split()
        
        bleu1 = sentence_bleu([gt1_tokens], pred1_tokens)
        bleu2 = sentence_bleu([gt2_tokens], pred2_tokens)
        bleu_scores.append((bleu1 + bleu2) / 2)
        
        # BERT Score 평가를 위해 전체 리스트에 추가
        preds_all.extend([pred1, pred2])
        refs_all.extend([gt1, gt2])
    
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    
    # BERT Score (F1) 계산 – 언어가 한국어인 경우 lang="ko" 사용
    P, R, F1 = bert_score(preds_all, refs_all, lang="ko", verbose=True)
    avg_bert_f1 = F1.mean().item()
    
    return avg_bleu, avg_bert_f1

def main():
    # device 설정: GPU가 있으면 사용, 없으면 CPU 사용
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 모델 및 토크나이저 로드 (모델 이름 또는 로컬 경로)
    model_name = "llama3.2-1B-instruct"  # 모델이 로컬에 설치되어 있어야 합니다.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    
    # JSON 파일에서 데이터 로드 (복합질의와 정답 단문질의 쌍)
    data = load_data("data.json")
    
    predictions = []
    ground_truths = []
    
    for item in data:
        complex_query = item["complex_query"]
        gt_simple = item["simple_queries"]  # [단문질의1, 단문질의2]
        
        # 모델을 통해 단문질의 생성
        pred1, pred2 = generate_simple_queries(complex_query, tokenizer, model, device)
        predictions.append((pred1, pred2))
        ground_truths.append((gt_simple[0], gt_simple[1]))
        
        # 진행 상황 출력
        print(f"복합질의: {complex_query}")
        print(f"정답 단문질의: {gt_simple}")
        print(f"생성 단문질의: {[pred1, pred2]}")
        print("------")
    
    # 평가: BLEU 및 BERT Score 계산
    avg_bleu, avg_bert_f1 = evaluate_predictions(predictions, ground_truths)
    print(f"평균 BLEU Score: {avg_bleu}")
    print(f"평균 BERT F1 Score: {avg_bert_f1}")

if __name__ == "__main__":
    main()
