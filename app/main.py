from dotenv import load_dotenv

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from agent import Agent

_ = load_dotenv()

# TODO: Datadogに変更する
from traceloop.sdk import Traceloop
Traceloop.init(disable_batch=True)

def main():
    st.title("Oracle Developer Days 2025")
    st.caption("実践！Datadogで高めるOCIのオブザーバビリティのデモで使用するアプリケーションです。")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input("どんなことを知りたいですか？"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            with st.spinner("考え中..."):
                # LLMの初期化
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17")
                agent = Agent(llm=llm)
                result = agent.run(prompt)
                st.markdown(result.content)
                st.session_state.messages.append({"role": "assistant", "content": result})

if __name__ == "__main__":
    main()
