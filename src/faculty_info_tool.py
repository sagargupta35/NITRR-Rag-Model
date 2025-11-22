from src.fs_utils import get_root_dir
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import AIMessage
from langchain.tools import tool

@tool
def faculty_info_tool(department: str, query: str) -> str:
    """
        Tool: faculty_info_tool

        Purpose:
        Use this tool to find specific information about faculty members from a single department at NIT Raipur.
        This is the correct tool for questions about faculty names, designations (like Professor or HOD),
        office numbers, email addresses, or other details stored in faculty records.

        Arguments:

        1. department (str):
        You MUST provide a single, valid department code. You must infer this from the user's query.
        (e.g., if the user asks about "computer science", you must use the code "cse").
        
        Valid department codes are:
        ['cse', 'it', 'electronics', 'electrical', 'mechanical', 'meta', 'chemical', 'civil', 'biotech', 'biomed']

        2. query (str):
        The specific, natural language question the user has about that department's faculty.
        *Example 1:* "Who is the Head of Department?"
        *Example 2:* "List all the Associate Professors."
        *Example 3:* "What is Dr. R. Gupta's office number and email?"

        Agent Strategy Hint (CRITICAL):

        This tool can only search ONE department at a time.
        If the user's query involves more than one department (e.g., "List the HODs for cse and it"),
        you MUST call this tool multiple times (once for 'cse' and once for 'it').
    """

    valid_departments = ['cse', 'it', 'electronics', 'electrical', 'mechanical', 'meta', 'chemical', 'civil', 'biotech', 'biomed']
    departement = department.lower()

    # check if department is valid
    if departement not in valid_departments:
        return f"department must be one of {valid_departments}"

    # load respective department data
    faculty_data_path = get_root_dir() / "data" / "faculty-data" / f"{departement}.json"
    try:
        faculty_data_string = faculty_data_path.read_text()
    except FileNotFoundError:
        return f"No data file found for department {departement}"
    except Exception as e:
        return f"unknown exception occurred: {e}. cannot fetch data file for department {department}"
    
    # create a chat model to answer the query
    llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0)

    system_prompt = """
        You are an academic assistant for National Institute of Technology Raipur.
        Your job is to answer the user's query using *only* the provided context.

        ### 1. Context & Domain
        The user's query is about faculty at NIT Raipur.
        The context provided to you is factual data retrieved from the institute's database (e.g., names, departments, designations, office numbers, emails).

        ### 2. Core Rules for Answering
        * **Strictly Factual:** You MUST base your entire answer *only* on the provided context.
        * **DO NOT** add, infer, or assume any information not explicitly stated in the context.
        * **Handle Missing Information:** If the context does not contain the information needed to answer the query, you MUST state: "That information is not available in the faculty records."
        * **No Hallucination:** It is better to state that the information is unavailable than to provide an incorrect or assumed answer.

        ### 3. Style Guide
        * **Tone:** Maintain a formal, professional, and helpful tone.
        * **Format:** If the context provides a list (e.g., multiple faculty members), format your answer as a bulleted list for clarity.
        * **Relevance:** Be concise and keep the answer strictly relevant to the user's query.
    """

    try:
        ai_message = llm.invoke(
            [
                {'role': 'system', 'content': system_prompt},
                {
                    'role': 'user',
                    'content': f"Based on the following faculty data, answer the given query:\nContext:\n{faculty_data_string}\n\nQuery:\n{query}"
                }
            ]
        )
    except Exception as e:
        return f"unknown exception occurred while querying llm to answer the query: {e}"

    response = _message_to_str(ai_message)

    return response


def _message_to_str(msg: AIMessage) -> str:
    response = ""

    # handle different msg.content type scenarios
    if isinstance(msg.content, str):
        response = msg.content.strip()
    elif isinstance(msg.content, list):
        response = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in msg.content
        ).strip()
    else:
        respose = str(msg.content)

    if not response:
        response = "Cannot answer the given query with the available context"

    return response


if __name__ == "__main__":
    from dotenv import load_dotenv

    # load the env file
    load_dotenv()

    while True:
        query = input(">> Enter your query: ")
        department = input(">> Enter the department: ")
        
        try:
            tool_message = faculty_info_tool.invoke({"department": department, "query": query})
            print(tool_message)
        except Exception as e:
            print(f"unknown exception occurred: {e}")


