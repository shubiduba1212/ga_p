from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import faiss
import numpy as np
import wikipedia
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인에서의 접근을 허용합니다.
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드를 허용합니다.
    allow_headers=["*"],  # 모든 헤더를 허용합니다.
)

# BART 모델 및 토크나이저 로드
model_name = 'facebook/bart-large'
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# 문서 데이터베이스 (위키백과에서 가져온 문서들)
documents = [
    get_wikipedia_content("Artificial Intelligence"),
    get_wikipedia_content("Natural Language Processing"),
    # 추가 문서들...
]

# 임베딩 생성
def create_embeddings(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model.get_encoder()(inputs['input_ids'])[0]
    return embeddings

# 문서 임베딩 생성
document_embeddings = create_embeddings(documents)
index = faiss.IndexFlatL2(document_embeddings.size(-1))
index.add(document_embeddings.numpy())

# Pydantic 모델 정의
class QueryRequest(BaseModel):
    query: str

@app.post("/generate")
async def generate_text(request: QueryRequest):
    query = request.query
    query_tokens = tokenizer(query, return_tensors='pt')
    
    with torch.no_grad():
        query_embedding = model.model.encoder(**query_tokens).last_hidden_state.mean(dim=1).numpy()

    k = 1
    distances, indices = index.search(query_embedding, k)
    
    if len(indices) == 0:
        raise HTTPException(status_code=404, detail="No relevant documents found.")

    retrieved_doc = documents[indices[0][0]]
    
    input_text = f"{retrieved_doc} {query}"
    input_tokens = tokenizer(input_text, return_tensors='pt')
    summary_ids = model.generate(input_tokens['input_ids'], num_beams=4, max_length=50, early_stopping=True)
    
    output_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return {"result": output_text}

# FastAPI 서버 실행 명령 (실행하려면 아래 코드 블록을 사용하지 말고 'uvicorn app:app --reload' 명령어를 사용하세요)
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
