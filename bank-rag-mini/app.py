import os
import re
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 讀取 .env 裡面的 OPENAI_API_KEY
load_dotenv()

DOCS_DIR = "data/sample_docs"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5")


def get_openai_client():
    """
    建立 OpenAI client。
    如果沒有設定 OPENAI_API_KEY，就回傳 None。
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return None

    return OpenAI(api_key=api_key)


def load_documents(docs_dir: str):
    """
    讀取 data/sample_docs 裡面的 txt 文件。
    """
    documents = []

    if not os.path.exists(docs_dir):
        return documents

    for filename in os.listdir(docs_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(docs_dir, filename)

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            documents.append({
                "filename": filename,
                "text": text
            })

    return documents


def split_text_into_chunks(text: str, filename: str):
    """
    把文件切成小段。
    這裡先用簡單規則：依照 1. 2. 3. 這種段落切分。
    """
    chunks = []

    parts = re.split(r"\n\s*\d+\.\s*", text)

    for i, part in enumerate(parts):
        clean_text = part.strip()

        if len(clean_text) > 20:
            chunks.append({
                "filename": filename,
                "chunk_id": i,
                "text": clean_text
            })

    return chunks


def build_chunks():
    """
    讀取所有文件，並切成 chunks。
    """
    documents = load_documents(DOCS_DIR)
    all_chunks = []

    for doc in documents:
        chunks = split_text_into_chunks(doc["text"], doc["filename"])
        all_chunks.extend(chunks)

    return all_chunks


def retrieve_relevant_chunks(question: str, chunks, top_k: int = 3):
    """
    使用 TF-IDF 找出跟問題最相似的段落。
    """
    if not chunks:
        return []

    texts = [chunk["text"] for chunk in chunks]

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts + [question])

    doc_vectors = vectors[:-1]
    question_vector = vectors[-1]

    similarities = cosine_similarity(question_vector, doc_vectors).flatten()

    ranked_indices = similarities.argsort()[::-1][:top_k]

    results = []
    for idx in ranked_indices:
        chunk = chunks[idx].copy()
        chunk["score"] = similarities[idx]
        results.append(chunk)

    return results


def build_context(retrieved_chunks):
    """
    把檢索到的 chunks 組成 LLM 可以閱讀的參考資料。
    """
    context_parts = []

    for chunk in retrieved_chunks:
        context_parts.append(
            f"""
[來源]
文件：{chunk["filename"]}
段落：Chunk {chunk["chunk_id"]}
相似度分數：{chunk["score"]:.4f}
內容：
{chunk["text"]}
"""
        )

    return "\n---\n".join(context_parts)


def generate_answer_with_llm(question: str, retrieved_chunks):
    """
    LLM 版本回答：
    1. 先檢查相似度
    2. 如果相似度太低，直接拒答
    3. 如果相似度足夠，把檢索到的段落交給 LLM
    4. 要求 LLM 只能根據來源回答
    """
    if not retrieved_chunks:
        return "目前知識庫中沒有任何可用文件，無法回答。"

    best_chunk = retrieved_chunks[0]

    # 門檻可以之後再調整
    if best_chunk["score"] < 0.05:
        return "目前知識庫中沒有足夠資料可以可靠回答這個問題。"

    client = get_openai_client()

    if client is None:
        return """
尚未設定 OPENAI_API_KEY，因此無法呼叫 LLM。

請在專案資料夾新增 .env，並填入：

OPENAI_API_KEY=你的 OpenAI API Key
OPENAI_MODEL=gpt-5.5
""".strip()

    context = build_context(retrieved_chunks)

    system_prompt = """
你是一位銀行內部知識查詢助理。

請嚴格遵守以下規則：
1. 只能根據「參考資料」回答。
2. 如果參考資料不足，請回答「目前知識庫中沒有足夠資料可以可靠回答這個問題」。
3. 不可以自行編造金融規定、利率、費用、流程或不存在的文件內容。
4. 回答要清楚、簡潔，適合銀行內部人員或客服人員閱讀。
5. 回答最後請列出你主要依據的來源文件與段落。
"""

    user_prompt = f"""
使用者問題：
{question}

參考資料：
{context}

請根據參考資料回答問題。
"""

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            instructions=system_prompt,
            input=user_prompt,
        )

        return response.output_text.strip()

    except Exception as e:
        return f"呼叫 LLM 時發生錯誤：{e}"


st.set_page_config(
    page_title="銀行內部知識查詢助理 Mini RAG + LLM",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 銀行內部知識查詢助理 Mini RAG + LLM")

st.write(
    "這是串接 LLM 的 Mini RAG Demo：先檢索金融文件，再讓 LLM 根據來源整理回答。"
)

chunks = build_chunks()

st.sidebar.header("系統狀態")
st.sidebar.write(f"已建立段落數：{len(chunks)}")
st.sidebar.write(f"目前模型：{OPENAI_MODEL}")

if get_openai_client() is None:
    st.sidebar.warning("尚未設定 OPENAI_API_KEY，LLM 功能無法使用。")
else:
    st.sidebar.success("已偵測到 OPENAI_API_KEY。")

with st.sidebar.expander("查看目前知識庫段落"):
    for chunk in chunks:
        st.write(f"**{chunk['filename']} - Chunk {chunk['chunk_id']}**")
        st.write(chunk["text"])
        st.divider()

question = st.text_input(
    "請輸入你的問題：",
    placeholder="例如：開戶時客戶沒有提供身分證明文件，客服應該怎麼處理？"
)

if st.button("送出問題"):
    if not question.strip():
        st.warning("請先輸入問題。")

    elif not chunks:
        st.error("目前沒有讀取到任何文件，請確認 data/sample_docs 裡面是否有 txt 檔案。")

    else:
        retrieved_chunks = retrieve_relevant_chunks(question, chunks, top_k=3)
        answer = generate_answer_with_llm(question, retrieved_chunks)

        st.subheader("AI 回答")
        st.write(answer)

        st.subheader("參考來源")
        for result in retrieved_chunks:
            st.markdown(
                f"""
**文件：** {result['filename']}  
**段落：** Chunk {result['chunk_id']}  
**相似度分數：** {result['score']:.4f}

> {result['text']}
"""
            )
            st.divider()