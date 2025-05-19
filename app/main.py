import os
from dotenv import load_dotenv

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from agent import Agent

_ = load_dotenv()
compartment_id = os.getenv("COMPARTMENT_ID", "ocid1.compartment.oc1..aaaaaaaanjtbllhqxcg67dq7em3vto2mvsbc6pbgk4pw6cx37afzk3tngmoa")

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
                # 最終的な要約タスクのみ出力トークンのコストパフォーマンスに優れたものを選択する
                summarize_llm = ChatOCIGenAI(
                    auth_type="INSTANCE_PRINCIPAL",
                    model_id="cohere.command-a-03-2025",
                    service_endpoint="https://inference.generativeai.ap-osaka-1.oci.oraclecloud.com",
                    compartment_id=compartment_id,
                    model_kwargs={
                        "max_tokens": 4_000
                    }
                )
                agent = Agent(llm=llm, summarize_llm=summarize_llm)
                result = agent.run(prompt)
                st.markdown(result.content)
                st.session_state.messages.append({"role": "assistant", "content": result})

if __name__ == "__main__":
    main()
