import os
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

def rag(query, page_db, chunk_db):
    """
    1. Retrieves relevant documents from Weaviate vector stores (page_db, chunk_db).
    2. Constructs a prompt with context.
    3. Sends the prompt to GPT-4 (via LangChain's ChatOpenAI).
    4. Returns the model's answer and the retrieved content for reference.
    """

    # Retrieve top documents
    page_docs = page_db.similarity_search(query, k=1)
    chunk_docs = chunk_db.similarity_search(query, k=1)

    # Combine text
    page_content = "\n---PAGE DOC---\n".join([doc.page_content for doc in page_docs])
    chunk_content = "\n---CHUNK DOC---\n".join([doc.page_content for doc in chunk_docs])

    # Combine metadata
    docs_metadata = "\n".join([
        f"{doc.metadata.get('source')} - Page {doc.metadata.get('page_number')}"
        for doc in page_docs + chunk_docs
    ])

    # System instruction (You can adapt this to your exact use-case/prompt style)
    system_prompt = (
        "Sen Türkçe konuşan bir yasal asistanısın. Kullanıcı soruyu sorduğunda, "
        "yalnızca sana verilen bağlamdaki bilgiyi kullanarak Türkçe yanıt ver. "
        "Eğer bilgiyi bağlamda bulamazsan, 'Bağlamda cevap bulunamadı.' şeklinde yanıt ver."
    )

    # Construct the final message for GPT-4
    context_prompt = f"""
### BAĞLAM
{chunk_content}

{page_content}

### SORU
{query}

### TALİMAT
- Sadece yukarıdaki bağlamdaki bilgiye dayanarak cevap ver.
- Bağlamda olmayan bilgiyi ekleme.
- Yanıtı Türkçe olarak ver.
"""

    # Initialize a ChatOpenAI with GPT-4
    llm = ChatOpenAI(
        model_name="gpt-4",  # or "gpt-4-32k", depending on your access
        temperature=0.1,       # or a higher temperature if you want more creative answers
        openai_api_key = api_key,
    )

    # Run the model with system + user messages
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=context_prompt),
    ]
    response = llm(messages)  # returns an AIMessage

    # The response text
    final_answer = response.content.strip()

    # Return final answer, retrieved documents text, and metadata
    # (so you can show them in the UI)
    return final_answer, (chunk_content + "\n\n" + page_content), docs_metadata
