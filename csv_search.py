import streamlit as st
import pandas as pd
from llm import ReportBuildingAgent
from dotenv import load_dotenv,find_dotenv
_ = load_dotenv(find_dotenv(), verbose=True, override=True)

st.set_page_config(page_title="SMART OFFER REPORT ASSISTANT", layout="wide")
st.title("SMART_OFFER_REPORT_ASSISTANT🤖")
st.markdown("Support natural language database search, mathematical calculations, and information summarization。")

# Initialize Agent (use cache to avoid repeated loading)
if "agent" not in st.session_state:
    st.session_state.agent = ReportBuildingAgent()
if "agent_history" not in st.session_state:
    st.session_state.agent_history = []
if "history" not in st.session_state:
    st.session_state.history = []
# Display session status in sidebar
with st.sidebar:
    st.header("SYSTEM_STATUS")
    if st.button("ClearHistory"):
        st.session_state.history = []
        st.session_state.agent_history = []
        st.rerun()
# Display chat history
for msg in st.session_state.history:

    with st.chat_message(msg["role"]):

        if msg["role"] == "user":
            st.write(msg["content"])

        else:

            intent = msg.get("intent")

            if intent:
                st.info(
                    f"IDENTIFY_INTENT: **{intent['intent_type']}** "
                    f"(CONFIDENCE: {intent['confidence']:.2f})"
                )

            st.subheader("ANALYZE_THE_RESULTS")
            st.write(msg["content"])

            sql_answer = msg.get("sql_answer")

            if isinstance(sql_answer, pd.DataFrame) and not sql_answer.empty:
                st.table(sql_answer)

            if msg.get("timestamp") and msg.get("answer_confidence"):
                st.caption(
                    f"ResponseTime: {msg['timestamp']} | "
                    f"AnswerReliability: {msg['answer_confidence']:.2f}"
                )
# Search input form
query = st.chat_input("Please enter your question (e.g. 'calculate 129*0.85', 'Retrieve all offers containing 'KFC''):")
if query:
    with st.chat_message("user"):
        st.write(query)
        # Cache conversation state
        st.session_state.history.append({
            "role": "user",
            "content": query
        })
if query:
    with st.spinner("THINKING..."):
        try:
            # Run LangGraph
            result = st.session_state.agent.run(query,st.session_state.agent_history)

            intent = result['intent']
            answer = result['final_answer']


            # Display intent recognition result
            st.info(f"IDENTIFY_INTENT: **{intent.intent_type}** (CONFIDENCE: {intent.confidence:.2f})")

            # Display final answer
            with st.chat_message("assistant"):

                st.subheader("ANALYZE_THE_RESULTS")
                st.write(answer.content)


                sql_answer = result.get("sql_answer")
                if isinstance(sql_answer, pd.DataFrame) and not sql_answer.empty:
                    st.table(sql_answer)

                # Display metadata
                st.caption(f"ResponseTime: {answer.timestamp} | AnswerReliability: {answer.confidence:.2f}")
                # Cache conversation state
                st.session_state.history.append({
                    "role": "assistant",
                    "content": answer.content if answer else "",
                    "intent": {
                        "intent_type": intent.intent_type,
                        "confidence": intent.confidence
                    } if intent else None,
                    "sql_answer": sql_answer,
                    "timestamp": answer.timestamp if answer else None,
                    "answer_confidence": answer.confidence if answer else None
                })

        except Exception as e:
            st.error(f"OperationError: {str(e)}")

# Example display
with st.expander("Example Queries"):
    st.write("- **Search**: Retrieve all offers containing 'KFC'")
    st.write("- **Math**: (500 - 120) × 0.9")
    st.write("- **Summary**: Provide a summary of the current offer data")