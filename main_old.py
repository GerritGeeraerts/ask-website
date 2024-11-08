import operator
from typing import TypedDict, Any, Dict, List
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END

from utils import llm, extract_protocol_and_domain, get_web_content

load_dotenv()

class AgentState(TypedDict):
    domain: str  # contains the base url of the website
    messages: Sequence[BaseMessage]  # contains initial messages from the user
    context_website: str  # what is the site about?
    question: str  # the question the user has
    current_url: str  # the url that is currently being checked
    current_url_content: str  # the content of the current url
    urls_done: Dict[Any, Any]  # TODO {"url": {"score": 0-10, "partial_answer": ""}}
    urls_queue: Dict[str, int]  # {"url": 0-10} probability score of url containing the answer
    final_answer : str  # the answer found
    max_url_loads: int # the maximum amount of urls that can be loaded
    step_logs: Annotated[str, operator.add]
    continue_searching_threshold : int
    continue_searching_increment : int

class WebQuestion(BaseModel):
  question: str = Field(description="The unanswered question the user has")
  url: str = Field(description="The url for the website where the answer is expected, do not include the path, "
                               "only protocol and domain. Example: https://sub.tweakers.net")

class Context(BaseModel):
    context: str = Field(description="What is this website about? Describe in about 50 words")

class WebAnswer(BaseModel):
    answer_candidate: str = Field(description="The complete or partial answer found on the page.")
    # likelyhood score of the answer being on a different page than this one.
    continue_searching_score: int = Field(
        description=("candidate answer to a question, can you reflect on the answers. What is the likelyhood that we "
                     "did not find a propper answer to the question because we did not look at enough subpages? Give a "
                     "score from 0 to 100 where 100 is that we deffently need to look at more webpages."
            # "Set this to True if the answer provided is incomplete, partial, ambiguous, "
            # "or not fully aligned with the user's query. The search should continue until "
            # "a thorough and clear response that fully addresses the user's question is found."
        )
    )

class Lead(BaseModel):
    url: str = Field(description="The url of where information may be found, or to an url that might contain an url to the answer")
    score: int = Field(description="The score of the url, 0-10: Where 10 is the highest probability of this url containing the answer")

class Leads(BaseModel):
    leads: Sequence[Lead] = Field(description="The leads to urls that might contain the answer")

def print_step(node: str, state: AgentState):
    if node == 'start_node':
        return (f"I am searching on {state['current_url']}:\n "
                f"{state['context_website']}\n"
                f"I am trying to answer this question:\n"
                f"{state['question']}")
    if node == 'load_content':
        return f"loading content for url:{state['current_url']}, (length={len(state['current_url_content'])})"
    if node == 'check_current_content_for_answer':
        if state.get('final_answer'):
            return f"Answer found on {state['current_url']}:\n{state['answer']}"
        return f"No answer found, continuing..."
    if node == 'check_current_content_for_leads':
        leads = [f"{k} [{v * 10}%] " for k, v in sorted(state.get('urls_queue', {}).items(), key=lambda item: item[1], reverse=True)]
        return f"Updated leads: {', '.join(leads)}"

def start_node(state: AgentState):
    """
    should extract the url and question from the chat history
    """
    result = llm.with_structured_output(WebQuestion).invoke(input=state["messages"])
    state["question"] = result.question
    state["urls_queue"] = {result.url: 10}
    state["current_url"] = result.url
    state['max_url_loads'] = state.get('max_url_loads', 10)
    state['domain'] = extract_protocol_and_domain(result.url)
    domain_content = get_web_content(state['domain'])
    context = llm.with_structured_output(Context).invoke(domain_content)
    state["context_website"] = context.context
    state["step_logs"] = f"{result.url}: {context}"
    state['continue_searching_threshold'] = state.get("continue_searching_threshold", 20)
    state['continue_searching_increment'] = state.get("continue_searching_increment", 5)
    return state


def load_content(state: AgentState):
    # order url_queue dictionaries by score
    state["urls_queue"] = dict(sorted(state["urls_queue"].items(), key=lambda item: item[1]))

    # get the highest scored url
    current_url, current_score = state["urls_queue"].popitem()
    state["current_url"] = current_url

    state["current_url_content"] = get_web_content(current_url)
    state['max_url_loads'] -= 1
    state['step_logs'] = f"loading: {current_url}"
    return state

def check_current_content_for_answer(state: AgentState):
    """
    should check highest scored url for the answer to the question
    """
    current_content = state["current_url_content"]
    messages = [
        HumanMessage(content="{c}\n\n---\n\n{q}\n\n---\n\n{context}\n\n{logs}".format(
            c=current_content,
            q=state["question"],
            context=state["context_website"],
            logs=state["step_logs"]
        ) # add all the previous steps taken to help guide the model of finding the good answer.
        )
    ]
    web_answer = llm.with_structured_output(WebAnswer).invoke(input=messages)

    if web_answer.continue_searching_score > state['continue_searching_threshold']:
        state['continue_searching_threshold'] += state['continue_searching_increment']
        state['step_logs'] = f"{web_answer.__dict}"
        return state  # continue search
    state["final_answer"] = web_answer.answer
    return state

def check_current_content_for_leads(state: AgentState):
    """
    should check last_url_checked_for_answer for url that could lead to the answer and grade them and them to
    the url_queue also reset the last_url_checked_for_answer and move back to check_url_for_answer
    """
    current_content = state["current_url_content"]
    message = "{c}\n\n---\n\nGive me the top 3 urls that could lead to the answer for the question: {q}".format(
        c=current_content, q=state["question"]
    )
    messages = [HumanMessage(content=message)]
    leads = llm.with_structured_output(Leads).invoke(input=messages)
    for lead in leads.leads:
        if lead.url in state["urls_queue"]:
            continue
        state["urls_queue"][lead.url] = lead.score
    state["urls_queue"] = dict(sorted(state["urls_queue"].items(), key=lambda item: item[1]))
    return state

def continue_to_search_for_leads(state: AgentState):
    if state.get("answer"):
        return END
    return "check_current_content_for_leads"

def continue_loading_new_page(state: AgentState):
    """
    If there are urls in the queue that have a score > 0 than return "check_url_for_answer" else END
    """
    if state['max_url_loads'] == 0:
        return END
    if any(state["urls_queue"].values()):
        return "load_content"
    return END

builder = StateGraph(AgentState)
builder.add_node("start_node", start_node)
builder.add_node("load_content", load_content)
builder.add_node("check_current_content_for_answer", check_current_content_for_answer)
builder.add_node("check_current_content_for_leads", check_current_content_for_leads)


builder.set_entry_point("start_node")
builder.add_edge("start_node", "load_content")
builder.add_edge("load_content", "check_current_content_for_answer")
builder.add_conditional_edges( # if answer is found go to END else go to check_url_for_leads
    "check_current_content_for_answer",
    continue_to_search_for_leads,
    {END: END, "check_current_content_for_leads": "check_current_content_for_leads"}
)
builder.add_conditional_edges( # if there are unexplored leads go to check_url_for_answer and not to many loads else END
    "check_current_content_for_leads",
    continue_loading_new_page,
    {END: END, "load_content": "load_content"}
)

graph = builder.compile()

for s in graph.stream(
        {"messages": [
            HumanMessage(content="What llm models offered by Groq have a context window larger then 200k tokens on https://groq.com/ ?"),
            # HumanMessage(content="Does https://www.skybad.de sell Pipe Insulation ?"),
        ]}
    ):
    for node, state in s.items():
        print(print_step(node, state))


# result = graph.invoke(
#     {"messages": [
#         HumanMessage(content="What llm models offered by Groq have a context window larger then 100k tokens on https://groq.com/ ?"),
#     ]}
# )