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

# 사용자 에이전트 설정
wikipedia.set_lang("ko")
wikipedia.set_user_agent("MyApp/1.0 (dldks1212@gmail.com)")

# Tokenizer 및 모델 로드
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")

# 문서 데이터베이스 초기화
documents = []
document_titles = []

def get_wikipedia_pages(query, max_pages=5):
    """위키백과에서 검색어로 관련된 페이지 목록 가져오기"""
    search_results = wikipedia.search(query, results=max_pages)
    return search_results

def get_wikipedia_content(title):
    """위키백과에서 특정 페이지의 내용을 가져오기"""
    try:
        page = wikipedia.page(title)
        return page.content
    except wikipedia.exceptions.DisambiguationError as e:
        # 여러 페이지가 발견된 경우 첫 번째 페이지 선택
        return wikipedia.page(e.options[0]).content
    except wikipedia.exceptions.PageError:
        return None

def update_document_database(query):
    """검색어와 관련된 위키백과 페이지 내용을 문서 데이터베이스에 추가하기"""
    titles = get_wikipedia_pages(query)
    for title in titles:
        content = get_wikipedia_content(title)
        if content:
            documents.append(content)
            document_titles.append(title)

# 문서 임베딩 생성 및 검색 인덱스 초기화
def create_faiss_index():
    """문서 임베딩을 생성하고 FAISS 인덱스 초기화"""
    global documents
    embeddings = []
    for doc in documents:
        inputs = tokenizer(doc, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model.encoder(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).numpy())
    embeddings = np.vstack(embeddings)
    
    global index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

# FastAPI 서버 및 엔드포인트 설정
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class Query(BaseModel):
    text: str

@app.post("/generate")
def generate_summary(query: Query):
    try:
        update_document_database(query.text)
        create_faiss_index()
        
        # 문서 검색
        inputs = tokenizer(query.text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model.encoder(**inputs)
            query_embedding = outputs.last_hidden_state.mean(dim=1).numpy()
        
        distances, indices = index.search(query_embedding, k=1)
        if len(indices[0]) > 0:
            relevant_document = documents[indices[0][0]]
            inputs = tokenizer(relevant_document, return_tensors="pt", padding=True, truncation=True)
            summary_ids = model.generate(inputs['input_ids'])
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return {"result": summary}
        else:
            return {"result": "No relevant document found."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# FastAPI 서버 실행 명령 (실행하려면 아래 코드 블록을 사용하지 말고 'uvicorn app:app --reload' 명령어를 사용하세요)
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
