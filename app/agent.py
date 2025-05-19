import os
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field
from typing import List

import oracledb

from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.language_models import BaseChatModel
from langchain_community.embeddings.oci_generative_ai import OCIGenAIEmbeddings
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langchain_community.vectorstores import OracleVS

from google.ai.generativelanguage_v1beta.types import Tool as GenAITool

# --- データ構造定義 ---
# LLMが構造化出力（structured_output）するためのPydanticモデル
class Session(BaseModel):
    title: str = Field(..., description="セッションのタイトル")
    abstract: str = Field(..., description="セッションの概要情報")
    tag: List[str] = Field(..., description="セッションの技術タグのリスト")

class Sessions(BaseModel):
    sessions: List[Session] = Field(..., default_factory=list, description="セッション情報のリスト")

class Recommendation(BaseModel):
    title: str = Field(..., description="学習コンテンツのタイトル")
    url: str = Field(..., description="学習コンテンツのURL")

class Recommendations(BaseModel):
    recommendations: List[Recommendation] = Field(..., default_factory=list, description="学習コンテンツのリスト")

class Evaluation(BaseModel):
    relevance_score: float = Field(..., description="関連性スコア")

# LangGraphで使用する状態管理用のモデル
class State(BaseModel):
    question: str = Field(..., description="ユーザーからの質問")
    structured_summary: Sessions = Field(
        default_factory=lambda: Sessions(sessions=[]),
        description="要約したセッション情報のリスト"
    )
    recommendation_summary: Recommendations = Field(
        default_factory=lambda: Recommendations(recommendations=[]),
        description="学習コンテンツのリスト"
    )
    relevance_score: float = Field(
        default=0.0,
        description="関連性スコア"
    )
    google_search_count: int = Field(
        default=0,
        description="Google検索の回数"
    )
    exective_summary: str = Field(
        default="",
        description="最終的な要約結果"
    )

# --- LangGraphのノード ---
class SessionRetriever:
    """
    Oracle Databaseからセッション情報を検索し、関連性の高いセッション情報を構造化データとして抽出するクラス。
    LangGraphのノードとして機能することを想定。
    """
    def __init__(self, llm: BaseChatModel):
        self.llm_structured = llm.with_structured_output(Sessions)
        _ = load_dotenv(find_dotenv())
        compartment_id = os.getenv("COMPARTMENT_ID")
        username = os.getenv("USERNAME")
        password = os.getenv("PASSWORD")
        dsn = os.getenv("DSN")
        self.retriever = self._get_retriever(
            compartment_id,
            username,
            password,
            dsn
        )
    
    def _get_retriever(self, compartment_id: str, username: str, password: str, dsn: str) -> VectorStoreRetriever:
        embedding_function = OCIGenAIEmbeddings(
            auth_type="INSTANCE_PRINCIPAL",
            service_endpoint="https://inference.generativeai.ap-osaka-1.oci.oraclecloud.com",
            model_id="cohere.embed-multilingual-v3.0",
            compartment_id=compartment_id,
        )
        connection = oracledb.connect(
            user=username,
            password=password,
            dsn=dsn
        )
        return OracleVS(
            client=connection,
            embedding_function=embedding_function,
            table_name="sessions"
        ).as_retriever(search_kwargs={"k": 5})
        
    def _format_docs(self, docs: List[Document]) -> str:
        formatted_docs = "\n\n".join(doc.page_content for doc in docs)
        print(f"{formatted_docs=}")
        return formatted_docs
        
    def run(self, state: State) -> State:
        retrieval_result = self.retriever.invoke(state.question)
        print(f"{retrieval_result=}")
        prompt = ChatPromptTemplate(
            messages = [
                SystemMessage(content="あなたは与えられた複数のセッション情報（検索結果）を注意深く分析し、ユーザーの質問に関連しそうなセッションを抽出します。抽出したセッションの情報を、与えられたテキストに基づいて、指定されたJSON形式（'sessions'キーの下に'title', 'abstract', 'tag'を持つオブジェクトのリスト）で出力してください。検索結果に含まれていない情報は決して生成してはいけません。技術タグ(tag)は、各セッションのabstractの内容から関連する技術要素を最大3つまで抽出してください。関連するセッションが見つからない場合は、'sessions'キーの値として空のリスト [] を含むJSONを出力してください。"),
                HumanMessagePromptTemplate.from_template("以下の検索結果の中から、ユーザーの質問に関連しそうなセッションの情報（タイトル、要約、タグ）をそのまま抜き出して、指示されたJSON形式で出力してください。検索結果に含まれないタイトルや要約を創作しないでください。\n\n ### ユーザーの質問 \n{question} \n\n ### 検索結果 \n{context}"),
            ]
        )
        chain = (
            {
                "question": RunnablePassthrough(),
                "context": self.retriever
            }
            | prompt
            | self.llm_structured # structured_output
        )
        result: Sessions = chain.invoke(state.question)
        state.structured_summary = result
        print(f"{state=}")
        return state

class Recommender:
    """
    Google検索を行い、関連セッションで事前に学習しておくべき内容を取得するクラス。
    LangGraphのノードとして機能することを想定。
    """
    def __init__(self, llm: BaseChatModel):
        self.llm_tool = llm.bind_tools(GenAITool(google_search={}))
        self.llm_tool = llm.with_structured_output(Recommendations)
    
    def _format_sessions(self, summarized_sessions: Sessions) -> str:
        formatted_sessions = "\n\n".join(
            f"タイトル： {session.title} \n概要： {session.abstract} \n技術タグ： {', '.join(session.tag)}"
            for session in summarized_sessions.sessions
        )
        print(f"{formatted_sessions=}")
        return formatted_sessions
    
    def run(self, state: State) -> State:
        prompt = ChatPromptTemplate(
            [
                SystemMessage(
                    content="あなたは与えられたセッション情報を分析し、ユーザーがそのセッションを参加する前に学んでおくべき内容をGoogle検索を行い、関連情報を取得します。関連情報は、その内容が一目でわかるようにタイトルとURLを含むJSON形式で出力してください。出力は、'recommendations'キーの下に'title'と'url'を持つオブジェクトのリストとして出力してください。関連情報が見つからない場合は、'recommendations'キーの値として空のリスト [] を含むJSONを出力してください。実際に検索によって得られていない情報は推奨リストに含めないでください。"
                ),
                HumanMessagePromptTemplate.from_template(
                    template="以下のセッション情報を分析し、ユーザーがそのセッションを参加する前に学んでおくべき内容をGoogle検索を行い、関連情報を取得してください。\n\n ### セッション情報 \n{structured_summary}\n\n"
                )
            ]
        )
        chain = (
            {"structured_summary": RunnableLambda(self._format_sessions)}
            | prompt
            | self.llm_tool # with tools and structured_output
        )
        result: Recommendations = chain.invoke(state.structured_summary)
        state.recommendation_summary = result
        # Google検索の回数をインクリメント
        state.google_search_count += 1
        print(f"{state=}")
        return state

class Evaluator:
    """
    Google検索結果とセッション情報を分析し、関連性を評価するクラス。
    LangGraphのノードとして機能することを想定。
    """
    def __init__(self, llm: BaseChatModel):
        self.llm = llm.with_structured_output(Evaluation)
    
    def run(self, state: State) -> State:
        prompt = ChatPromptTemplate(
            [
                SystemMessage(
                    content="あなたは与えられたセッション情報とGoogle検索結果を分析し、その関連度を評価します。関連度は0から1の範囲で、1が最も関連性が高いことを示します。出力は、'relevance_score'キーの下に関連度スコアを持つJSON形式で出力してください。関連度スコアが0の場合は、'relevance_score'キーの値として0.0を含むJSONを出力してください。"
                ),
                HumanMessagePromptTemplate.from_template(
                    template="あなたは与えられたセッション情報とGoogle検索結果を分析し、その関連度を評価してください。\n\n ### セッション情報 \n{structured_summary}\n\n ### Google検索結果 \n{recommendation_summary}\n\n"
                )
            ]
        )
        chain = (
            {"structured_summary": RunnablePassthrough(), "recommendation_summary": RunnablePassthrough()}
            | prompt
            | self.llm # with tools and structured_output
        )
        result: Evaluation = chain.invoke(
            {"structured_summary": state.structured_summary, "recommendation_summary": state.recommendation_summary}
        )
        state.relevance_score = result.relevance_score
        print(f"{state=}")
        return state

class Summarizer:
    """
    一連の情報から最終的な要約を生成するクラス。
    LangGraphのノードとして機能することを想定。
    最終的な出力は、LLM自体の性能はそこまで不要なため価格重視でOCIを利用する。
    """
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
    
    def _format_structured_summary(self, structured_summary: Sessions) -> str:
        formatted_summary = "\n\n".join(
            f"タイトル： {session.title} \n概要： {session.abstract} \n技術タグ： {', '.join(session.tag)}"
            for session in structured_summary.sessions
        )
        print(f"{formatted_summary=}")
        return formatted_summary
    
    def _format_recommendation_summary(self, recommendation_summary: Recommendations) -> str:
        formatted_summary = "\n\n".join(
            f"タイトル： {recommendation.title} \nURL： {recommendation.url}"
            for recommendation in recommendation_summary.recommendations
        )
        print(f"{formatted_summary=}")
        return formatted_summary
        
    def run(self, state: State) -> State:
        prompt = ChatPromptTemplate(
            [
                SystemMessage(
                    content="あなたは与えられたセッション情報とGoogle検索結果を分析し、ユーザーがそのセッションを参加する前に学んでおくべき内容や見どころを要約します。最終的な要約は、以下の形式で出力してください。\n ### セッションタイトル \n<セッションタイトル>\n\n ### セッション概要 \n<セッション概要>\n\n ### 学習コンテンツ \n<学習コンテンツのタイトルとURLのリスト>\n\n ### 見どころ \n<見どころ>"
                ),
                HumanMessagePromptTemplate.from_template(
                    template="以下のセッション情報とGoogle検索結果を分析し、最終的な要約を作成してください。\n\n ### セッション情報 \n{structured_summary}\n\n ### Google検索結果 \n{recommendation_summary}"
                )
            ]
        )
        chain = (
            {
                "structured_summary": RunnablePassthrough(),
                "recommendation_summary": RunnablePassthrough()
            }
            | prompt
            | self.llm
        )
        result: str = chain.invoke({"structured_summary": state.structured_summary, "recommendation_summary": state.recommendation_summary})
        state.exective_summary = result
        print(f"{state=}")
        return state

# --- エージェント実装 ---
class Agent:
    def __init__(self, llm: BaseChatModel, summarize_llm: BaseChatModel = None):
        self.llm = llm
        # 各ツールの初期化
        self.session_retriever = SessionRetriever(llm=self.llm)
        self.recommender = Recommender(llm=self.llm)
        self.evaluator = Evaluator(llm=self.llm)
        if summarize_llm == None:
            self.summarize_llm = llm
        else:
            self.summarize_llm = summarize_llm
        self.summarizer = Summarizer(llm=self.summarize_llm)
        self.graph = self._get_compiple_graph()
    
    
    def _get_compiple_graph(self) -> CompiledStateGraph:
        # StateGraphの初期化
        graph_builder = StateGraph(state_schema=State)
        # ノードの追加
        graph_builder.add_node(
            node="session_retriever",
            action=self.session_retriever.run,
        )
        graph_builder.add_node(
            node="recommender",
            action=self.recommender.run
        )
        graph_builder.add_node(
            node="evaluator",
            action=self.evaluator.run
        )
        graph_builder.add_node(
            node="summarizer",
            action=self.summarizer.run,
        )
        # エッジ（ノード間の繋がり）の追加
        graph_builder.add_edge("session_retriever", "recommender")
        graph_builder.add_edge("recommender", "evaluator")
        # 条件付きエッジの追加（再度Google検索するか、要約するか）
        graph_builder.add_conditional_edges(
            "evaluator",
            # Google検索の回数が5回を超えるか、関連性スコアが0.5を超える場合は、最終出力へ
            lambda state: state.google_search_count > 5 or state.relevance_score > 0.5,
            {True: "summarizer", False: "recommender"}
        )
        # 始点と終点の設定
        graph_builder.set_entry_point("session_retriever")
        graph_builder.set_finish_point("summarizer")
        graph = graph_builder.compile()
        return graph
    
    def run(self, question: str) -> str:
        initial_state = State(question=question)
        print(f"{initial_state=}")
        final_state = self.graph.invoke(initial_state)
        print(f"{final_state=}")
        return final_state["exective_summary"]
