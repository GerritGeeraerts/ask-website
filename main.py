import operator
import os
from pprint import pprint
from pyexpat.errors import messages
from typing import TypedDict, Any, Dict
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from firecrawl import FirecrawlApp

from langchain_aws import ChatBedrock

from langgraph.graph import StateGraph, END

from utils import llm

load_dotenv()

class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    question: str
    # current_url: str
    # current_content: str
    # explored_urls: Dict[str, Dict[str: Any] # {"url": {"score": 0-10, "partial_answer": ""}}
    url_queue: Dict[str, int] # {"url": 0-10} probability score use -1 to mark as done
    answer : str
    content_last_checked_for_answer: str
    # max_url_loads: int

class WebQuestion(BaseModel):
  question: str = Field(description="The unanswered question the user has")
  url: str = Field(description="The url for the website where the answer is expected, do not include the path, "
                               "only protocol and domain. Example: https://sub.tweakers.net")

class WebAnswer(BaseModel):
    answer: str = Field(description="The answer to the question")
    continue_searching: bool = Field(description="If the answer is vague or not exactly what the user is looking for, set this to True")

class Lead(BaseModel):
    url: str = Field(description="The url of where information may be found, or to an url that might contain an url to the answer")
    score: int = Field(description="The score of the url, 0-10: Where 10 is the highest probability of this url containing the answer")

class Leads(BaseModel):
    leads: Sequence[Lead] = Field(description="The leads to urls that might contain the answer")

def print_state(state: AgentState):
    state_copy = state.copy()
    state_copy.pop('messages') if 'messages' in state_copy else None
    state_copy.pop('content_last_checked_for_answer') if 'content_last_checked_for_answer' in state_copy else None
    # pprint(state_copy)

def start_node(state: AgentState):
    """
    should extract the url and question from the chat history
    """
    result = llm.with_structured_output(WebQuestion).invoke(input=state["messages"])
    state["question"] = result.question
    state["url_queue"] = {result.url: 10}
    print_state(state)
    return state

def check_url_for_answer(state: AgentState):
    """
    should check highest scored url for the answer to the question
    """
    print('check_url_for_answer')
    unexplored_urls = {k: v for k, v in state["url_queue"].items() if v > 0}
    most_promising_url = max(unexplored_urls, key=unexplored_urls.get)
    app = FirecrawlApp(api_key=os.environ.get('FIRECRAWL_API_KEY'))
    response = app.scrape_url(url=most_promising_url, params={
        'formats': ['markdown'],
        'onlyMainContent': False
    })
    state["content_last_checked_for_answer"] = response['markdown']
    messages = [
        HumanMessage(content="{c}\n\n---\n\n{q}".format(c=response['markdown'], q=state["question"]))
    ]
    web_answer = llm.with_structured_output(WebAnswer).invoke(input=messages)
    state['url_queue'][most_promising_url] = -1
    if web_answer.continue_searching:
        print_state(state)
        return state
    state["answer"] = web_answer.answer
    print_state(state)
    return state

def check_url_for_leads(state: AgentState):
    """
    should check last_url_checked_for_answer for url that could lead to the answer and grade them and them to
    the url_queue also reset the last_url_checked_for_answer and move back to check_url_for_answer
    """
    print("check_url_for_leads")
    content_last_checked_for_answer = state["content_last_checked_for_answer"]
    messages = [
        HumanMessage(content="{c}\n\n---\n\nGive me the top 3 urls that could lead to the answer for the question: {q}".format(c=content_last_checked_for_answer, q=state["question"]))
    ]
    leads = llm.with_structured_output(Leads).invoke(input=messages)
    for lead in leads.leads:
        if lead.url in state["url_queue"]:
            continue
        state["url_queue"][lead.url] = lead.score
    print_state(state)
    return state

def answer_found_or_search_leads(state: AgentState):
    if state.get("answer"):
        return END
    return "check_url_for_leads"

def unexplored_leads(state: AgentState):
    """
    If there are urls in the queue that have a score > 0 than return "check_url_for_answer" else END
    """
    if any(state["url_queue"].values()):
        return "check_url_for_answer"
    return END

builder = StateGraph(AgentState)
builder.add_node("start_node", start_node)
builder.add_node("check_url_for_answer", check_url_for_answer)
builder.add_node("check_url_for_leads", check_url_for_leads)


builder.set_entry_point("start_node")
builder.add_edge("start_node", "check_url_for_answer")
builder.add_conditional_edges( # if answer is found go to END else go to check_url_for_leads
    "check_url_for_answer",
    answer_found_or_search_leads,
    {END: END, "check_url_for_leads": "check_url_for_leads"}
)
builder.add_conditional_edges( # if there are unexplored leads go to check_url_for_answer else END
    "check_url_for_leads",
    unexplored_leads,
    {END: END, "check_url_for_answer": "check_url_for_answer"}
)

graph = builder.compile()

for s in graph.stream(
        {"messages": [
            HumanMessage(content="What llm models offered by Groq have a context window larger then 100k tokens on https://groq.com/ ?"),
        ]}
    ):
    for node, node_message in s.items():
        print(node)


# result = graph.invoke(
#     {"messages": [
#         HumanMessage(content="What llm models offered by Groq have a context window larger then 100k tokens on https://groq.com/ ?"),
#     ]}
# )