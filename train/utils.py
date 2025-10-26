import hashlib, numpy as np
import time
import numpy as np
import torch
from transformers import set_seed as hf_set_seed

def set_seeds(seed: int):
    """
    torch, numpy, Hugging Face(Transformers)에 동일 시드를 설정합니다.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)
    print(f"[Seed set] {seed}")

def make_seed(auto: bool = False, default: int = 42) -> int:
    """
    auto=True면 실행마다 다른 시드(시간 기반), False면 고정 시드 사용.
    """
    return (int(time.time() * 1000) & 0xFFFFFFFF) if auto else default

def model_checksum(m):
    s = ""
    for n,p in m.named_parameters():
        s += str(np.sum(p.detach().cpu().numpy()))
    return hashlib.md5(s.encode()).hexdigest()

def add_prompt(text, prompt):
    return f"{prompt} {text}" if prompt else text

# def add_prompt(text: str, prompt: str) -> str:

#     text = (text or "").strip()
#     prompt = prompt or ""

#     # 문서 템플릿(title/text) 처리
#     if "title:" in prompt and "text:" in prompt:
#         if "title: none" in prompt:
#             # 제목만 있을 때: title 채우고 text는 비워둠
#             return prompt.replace("title: none", f"title: {text}", 1)
#         # title이 이미 채워진 템플릿이면 text 뒤에 본문을 붙임
#         sep = "" if prompt.endswith((" ", ":", "\t")) else " "
#         return f"{prompt}{sep}{text}"

#     # 일반 프롬프트 처리
#     sep = "" if prompt.endswith((" ", ":", "\t")) else " "
#     return f"{prompt}{sep}{text}"