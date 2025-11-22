from langchain.tools import tool
from src.vector_store import get_collection
from langchain_google_genai import ChatGoogleGenerativeAI
from rich.console import Console
from rich.markdown import Markdown

@tool
def ordinance_tool(query: str, filters: dict = {}) -> str:
    """
        Tool: ofaculty_tool
        
        Purpose: 
        Use this tool to find official rules, policies, and regulations from the NIT Raipur Ordinance documents. 
        It is the best tool for any question related to academic rules, grading, credits, or policies.
        
        Input Fields:
        
        1. search_query (string):
           A concise semantic search query, derived from the user's question.
           Example: If the user asks, "How is the SPI for B.Tech students calculated?", the search_query should be "SPI calculation for B.Tech".
        
        2. metadata_filter (dict):
           A mandatory ChromaDB filter dictionary to narrow the search. You must infer the correct filter from the query.
        
        Metadata & Filter Rules:
        
        Available Filter Fields:
        * degree (string): Valid values are "B.tech", "B.Arch", "M.Tech", "MCA", "PHD" OR "M.SC".
        * program_level (string): Valid values are "Undergraduate" or "Postgraduate".
        
        Filter Syntax (Strictly Enforced):
        You MUST generate a valid Chroma filter dictionary.
        
        * For a single condition: Use the field name as the top-level key.
            Example: {"degree": {"$eq": "B.Tech"}}
        
        * For multiple conditions: You MUST use a single top-level $and operator.
            Example: {"$and": [{"degree": {"$eq": "B.Tech"}}, {"program_level": {"$eq": "Undergraduate"}}]}
        
        CRITICAL ERROR TO AVOID:
        Never use multiple fields at the top level. You must use conjunction operators like "and", "or".
        * WRONG: {"degree": "B.Tech", "program_level": "Undergraduate"}
        * WRONG: {{"degree": {"$eq": "B.Tech"}}, {"program_level": {"$eq": "Undergraduate"}}}
        * RIGHT: {"$and": [{"degree": {"$eq": "B.Tech"}}, {"program_level": {"$eq": "Undergraduate"}}]}
        
        Agent Strategy Hint:
        
        If a user's query asks to compare two different domains (e.g., "How is SPI calculated for B.Tech and B.Arch students?"), do not use an $or filter.
        
        Instead, call this tool multiple times (once for B.Tech, once for B.Arch). This ensures you retrieve the distinct, correct context for each program, which is essential for an accurate answer.
        """

    ordinance_collection = get_collection('ordinance')

    if filters:
        results = ordinance_collection.query(query_texts=[query], n_results=5, where=filters)
    else:
        results = ordinance_collection.query(query_texts=[query], n_results=3)

    docs = results["documents"][0] # type: ignore
    metas = results["metadatas"][0] # type: ignore

    # Combine results for LLM
    res =  "\n\n".join(
        f"Source: {meta.get('source')} (Page {meta.get('page', '?')})\n{doc}"
        for doc, meta in zip(docs, metas)
    )
    return res


if __name__ == "__main__":

    from dotenv import load_dotenv

    load_dotenv()

    vector_llm_system_prompt = """
        System Prompt: NIT Raipur Academic Assistant

        You are an academic assistant for National Institute of Technology Raipur. Your primary function is to provide students with factual and clear information regarding academic regulations and policies.

        1. Core Directive: Use the Ordinance Tool

        Your most important task is to retrieve information.

        * You MUST use the `OrdinanceRetriever` tool for *any* query that may relate to institute rules, regulations, or policies.
        * This includes, but is not limited to: admission, grading, SPI/CPI, attendance, credits, examinations, degree requirements, and academic penalties.

        2. Rules for Answering

        After you receive context from the tool, follow these rules strictly:

        * Be 100% Factual: Base your entire answer *only* on the retrieved context. DO NOT add any information, assumptions, or external knowledge, even if it seems helpful.
        * Cite Your Source: You MUST conclude your answer by citing the source and page number from the metadata (e.g., "*(Source: Academic Ordinance 2024, Page 15)*").
        * Handle Missing Information: If the retrieved context does not contain the answer or is ambiguous, you MUST state: "The provided ordinance does not specify this clearly."
        * Handle Multiple Rules: If the context provides different rules (e.g., for B.Tech vs. M.Tech), clearly explain the conditions for each.

        3. Style Guide

        * Tone: Maintain a formal, professional, and helpful student-facing tone.
        * Clarity: Use simple, direct language. Avoid jargon.
        * Format: Prefer short paragraphs and bullet points for scannability.
        * Relevance: Keep your answer strictly focused on the user's specific query. Do not add irrelevant details.

        4. Summary of Your Workflow

        1.  Analyze Query: Does this question *sound* like it's about a rule?
        2.  Call Tool: If yes, formulate a query and call `OrdinanceRetriever`.
        3.  If not, answer it by yourself if you feel confident. Otherwise give a clear response that you are not able to answer that with your current knowledge.
        3.  Read Context: Carefully analyze the text and metadata returned by the tool.
        4.  Synthesize Answer: Compose a factual answer based *only* on the context, following all style and citation rules.
    """
    
    model_with_tools = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite").bind_tools([ordinance_tool])


    messages = [{'role': 'system', 'content': vector_llm_system_prompt}]

    while True:
        query = input("enter your query: ")
        messages.append({'role': 'user', 'content': query})

        ai_msg = model_with_tools.invoke(messages)
        messages.append(ai_msg) # type: ignore
        
        import uuid

        for call in ai_msg.tool_calls:
            call_id = call.get("id", str(uuid.uuid4()))
            result = ordinance_tool.invoke(call)
            messages.append({
                'role': 'tool',
                'tool_call_id': call_id, # type: ignore
                'content': result
            })

        final = model_with_tools.invoke(messages)
        messages.append(final) # type: ignore

        console = Console()
        try:
            console.print(Markdown(final.content)) # type: ignore
        except Exception:
            print(final.content)

