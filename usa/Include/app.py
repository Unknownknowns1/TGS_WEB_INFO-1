import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

load_dotenv()

url="https://en.wikipedia.org/wiki/LangChain"


documents = WebBaseLoader(url).load()
website_text=""

for d in documents:
    website_text=website_text+d.page_content

class llm(BaseChatModel):
    """A concrete chat model that delegates generation to ChatGroq."""

    client: ChatGroq = Field(default_factory=lambda: ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY"),
    ))

    @property
    def _llm_type(self) -> str:
        return "groq-website-chat"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        response = self.client.invoke(messages, stop=stop, **kwargs)
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=response.content))]
        )


llm = llm()


prompt = ChatPromptTemplate.from_template("""
you are helpful assistant Answer only from the Given Website content.

if the answer is not in the given website,reply:
i dont have that information in the website
Website_content:{website_text}
Question:{question}
""")
chain=prompt|llm