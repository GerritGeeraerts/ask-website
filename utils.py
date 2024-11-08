from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

#
llm = ChatBedrock(
    credentials_profile_name="default", model_id="anthropic.claude-3-5-sonnet-20240620-v1:0"
)

load_dotenv()

# llm = ChatGroq(
#     model="llama-3.1-70b-versatile",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
# )

# llm = ChatOpenAI(
#     model="gpt-4o",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
# )