import os
import sqlite3
import re
from datetime import datetime
from typing import Annotated, Literal, TypedDict, List, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field, validator, ConfigDict
from langchain_openai import AzureChatOpenAI, OpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.tools import tool
from langchain_community.utilities import SQLDatabase
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv,find_dotenv
from simpleeval import simple_eval
_ = load_dotenv(find_dotenv(), verbose=True, override=True)


# --- 1. Pydantic Schemas (Data validation) ---

class UserIntent(BaseModel):
    """Schema for identifying user intent"""
    intent_type: Literal['qa', 'calculation', 'summarization', 'general'] = Field(description="Type of the user query")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score of the intent classification")
    reasoning: str = Field(description="Reason for the intent classification")


class AnswerResponse(BaseModel):
    """Schema for the final output"""
    content: str = Field(description="Response content")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence of the answer")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


# --- 2. State definition ---

class AgentState(TypedDict):

    query: str
    intent: Optional[UserIntent]
    sql_results: Optional[str]
    tool_output: Optional[str]
    sql_answer: Optional[pd.DataFrame]
    final_answer: Optional[AnswerResponse]
    history: List[dict]

# --- 3. Tools (Tool implementation) ---

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Only numbers and basic operators (+-*/().) are allowed."""
    # Security check: only allow mathematical characters
    if not re.match(r"^[0-9+\-*/().\s]+$", expression):
        return "Error: Expression contains illegal characters. For safety, only basic operations are supported."

    try:
        # Additional filtering before using eval (in production simpleeval is recommended)
        result = simple_eval(expression, {"__builtins__": None}, {})
        return str(result)
    except Exception as e:
        return f"Calculation error: {str(e)}"


# --- 4. Core class implementation ---

class ReportBuildingAgent:
    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.llm = AzureChatOpenAI(
            azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            temperature=0
        )
        self.db = SQLDatabase.from_uri("sqlite:///offer_db.sqlite")
        self.graph = self._build_graph()
        graph_png = self.graph.get_graph().draw_mermaid_png()
        with open("csv_searcher.png", "wb") as f:
            f.write(graph_png)
        #self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        self.embeddings = AzureOpenAIEmbeddings(azure_deployment=os.environ["text-embedding_3_large_deployment"],api_version=os.environ["text-embedding_3_large_api_version"])

    def format_history_for_llm(self,history):

        messages = []

        for h in history:

            # user message
            messages.append({
                "role": "user",
                "content": h["query"]
            })

            # assistant message
            assistant_parts = []

            if h.get("intent"):
                assistant_parts.append(
                    f"Intent: {h['intent'].intent_type} (confidence={h['intent'].confidence})"
                )

            if h.get("sql_results"):
                assistant_parts.append(
                    f"SQL Results: {h['sql_results']}"
                )

            if h.get("tool_output"):
                assistant_parts.append(
                    f"Tool Output: {h['tool_output']}"
                )

            if h.get("final_answer"):
                assistant_parts.append(
                    f"Answer: {h['final_answer'].content}"
                )

            assistant_message = "\n".join(assistant_parts)

            messages.append({
                "role": "assistant",
                "content": assistant_message
            })

        return messages

    def parse_output(self, retrieved_offers, query):
        from langchain_community.vectorstores import FAISS
        """Parse the output of retrieve_offers() and return a DataFrame using FAISS similarity_search"""

        # Split retrieved offers
        top_offers = retrieved_offers.split("#")
        vector_db = FAISS.from_texts(texts=top_offers, embedding=self.embeddings)
        docs_and_scores = vector_db.similarity_search_with_score(query, k=len(top_offers))

        # Construct DataFrame
        # In FAISS, score is typically L2 distance (smaller means more similar)
        # If inner product is used, larger values indicate higher similarity
        df = pd.DataFrame([
            {"distanceScore %": score, "offer": doc.page_content}
            for doc, score in docs_and_scores
        ])

        df.index += 1
        return df


    def _get_chat_prompt_template(self, intent: str) -> ChatPromptTemplate:
        """Dynamically select prompt template based on intent"""
        templates = {
            "qa": "You are a data analyst. Answer the user's question about the offer based on the SQL query results. Result: {context}",
            "summarization": "You are a refined assistant. Please summarize the following offer information and highlight the key offers：{context}",
            "calculation": "You are a math expert. Please explain the calculation process and give the result：{context}",
            "general": "You are a friendly assistant. Please answer the user's question。"
        }
        sys_msg = templates.get(intent, templates["general"])
        return ChatPromptTemplate.from_messages([
            ("system", sys_msg),
            ("placeholder", "{history}"),
            ("human", "{query}"),
        ])

    # --- Node functions ---

    def intent_classifier(self, state: AgentState):
        """Intent classification node"""
        structured_llm = self.llm.with_structured_output(UserIntent)
        system_prompt = "Analyze and categorize user queries: 'qa' (query database), 'calculation' (mathematical calculation), 'summarization' (summarization information), 'general' (other)."
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("placeholder", "{history}"),
            ("human", "{query}")
        ])
        llm_history = self.format_history_for_llm(state["history"])
        intent = structured_llm.invoke(prompt.format(query=state['query'],history=llm_history))
        return {"intent": intent}

    def sql_search_node(self, state: AgentState):
        """Database retrieval node"""
        PROMPT_TEMPLATE = """
                        You receive a query and your task is to retrieve the relevant offer from the 'OFFER' field in the 'offer_retailer' table.
                        Queries can be mixed case, so search for the uppercase version of the query as well.
                        Importantly, you may need to use information from other tables in the database, i.e.: 'brand_category', 'categories', 'offer_retailer', to retrieve the correct offer.
                        Don't make up offers. If you can't find an offer in the 'offer_retailer' table, return the string: 'NONE'.
                        If you can retrieve offers from the 'offer_retailer' table, separate each offer with the separator '#'. For example, the output should look something like this: 'offer1#offer2#offer3'.
                        If SQLResult is empty, return 'None'. Do not generate any offers.
                        Don't return any Markdown formatting, don't start or end with ''.
                        Only plain text SQL statements are returned.

                        This is the query: '{}'
                        """
        sqlichain_prompt=PROMPT_TEMPLATE.format(state['query'])

        # Integrate the original SQL generation logic here
        from langchain_experimental.sql import SQLDatabaseChain
        db_chain = SQLDatabaseChain.from_llm(self.llm, self.db)
        try:
            res = db_chain.run(sqlichain_prompt)
            df=self.parse_output(res, state['query'])
            return {"sql_results": res,"sql_answer":df}
        except Exception as e:
            return {"sql_results": f"Query failed: {str(e)}"}

    def calculator_node(self, state: AgentState):
        """Calculator processing node"""
        # Extract expression (simple approach; in practice LLM can extract it)
        expr = state['query']
        result = calculator.invoke(expr)
        return {"tool_output": result}

    def final_generator(self, state: AgentState):
        """Final answer generation node"""
        intent_type = state['intent'].intent_type
        context = state.get('sql_results') or state.get('tool_output') or ""

        prompt_tmpl = self._get_chat_prompt_template(intent_type)
        chain = prompt_tmpl | self.llm.with_structured_output(AnswerResponse)
        llm_history = self.format_history_for_llm(state["history"])
        response = chain.invoke({"query": state['query'], "context": context,"history": llm_history})
        return {"final_answer": response}


    # --- Graph construction ---

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("classify", self.intent_classifier)
        workflow.add_node("retrieve", self.sql_search_node)
        workflow.add_node("calculate", self.calculator_node)
        workflow.add_node("generate", self.final_generator)
        # workflow.add_node("memory", self.update_memory)

        # Set entry point
        workflow.set_entry_point("classify")

        # Routing logic (Conditional Edges)
        def route_by_intent(state: AgentState):
            it = state['intent'].intent_type
            if it == "calculation": return "calculate"
            if it == "qa" or it == "summarization": return "retrieve"
            return "generate"

        workflow.add_conditional_edges(
            "classify",
            route_by_intent,
            {
                "calculate": "calculate",
                "retrieve": "retrieve",
                "generate": "generate"
            }
        )

        workflow.add_edge("calculate", "generate")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()


    def run(self, query: str,agent_history=None):
        if agent_history is None:
            agent_history = []

            # query cache
        for h in agent_history:
            if h.get("query") == query:
                return h
        result = self.graph.invoke({
            "query": query,
            "history": agent_history
        })
        agent_history.append(result)
        return result