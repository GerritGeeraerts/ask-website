import operator
from pprint import pprint
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.constants import END
from langgraph.graph import StateGraph

from schemas import Question, Answer
from utils import llm, extract_protocol_and_domain, get_web_content

load_dotenv()

class AgentState(TypedDict):
    domain: str  # contains the base url of the website
    messages: list  # contains initial messages from the user
    question: str  # the question the user has
    urls_done: dict  # {"url": {"continue_searching_score": 0-100, "partial_answer": "a partial answer if there is one"}}
    urls_queue: dict  # {"url": 0-10} probability score of url containing the answer
    final_answer : str  # the answer found
    continue_searching_threshold : int
    continue_searching_increment : int

def start_node(state: AgentState):
    # preparing agent state so the rest of the code can be generic
    question = llm.with_structured_output(Question).invoke(state['messages'])
    state['question'] = question.question
    state['domain'] = extract_protocol_and_domain(question.url)
    state['urls_queue'] = {state['domain']: 101} # add domain as priority
    state['urls_queue'].update({question.url: 102}) # add question url as top priority (can override the domain url)
    state['urls_done'] = {}
    state['continue_searching_threshold'] = 10  # if the score is above this threshold, continue searching
    state['continue_searching_increment'] = 5  # each page that is visited, the threshold is increased by this amount
    return state

def extract_data(state: AgentState):
    url, score = state["urls_queue"].popitem()
    content = get_web_content(url)
    augment_web_content = (f'current url:{url}\n\n---\n'
                           f'current content:\n{content}\n\n---\n'
                           f'context_of_website:\n{state.get("context_website", "no context provided yet")}\n\n---\n '
                           f'previous_urls:\n{state.get("urls_done", "no previous steps")}\n\n---\n'
                           f'question:\n{state["question"]}\n\n---\n'
                           f'if you want to extracting final_answer you need at least a continue_searching_threshold '
                           f'of {state["continue_searching_threshold"]}. Also take into account previous answer '
                           f'candidates.')

    answer :Answer = llm.with_structured_output(Answer).invoke(augment_web_content)

    # Add answer as partial answer to urls_done
    state['urls_done'][url] = {
        "partial_answer": answer.answer_candidate,
        "continue_searching_score": answer.continue_searching_score
    }

    # check if we can stop searching
    if any(sub_dict['continue_searching_score'] < state['continue_searching_threshold'] for sub_dict in state['urls_done'].values()):
        if not answer.final_answer:
            raise ValueError("No final answer found, but continue_searching_threshold is reached")
        urls_visited = ', '.join(list(state['urls_done'].keys())[:-1])
        concluded_url = list(state['urls_done'].keys())[-1]
        state['final_answer'] = (f"Looking at {urls_visited} i concluded my answer on {concluded_url}\n"
                                 f"{answer.final_answer}")
        state['messages'].append(AIMessage(content=state['final_answer']))
        return state

    # update new leads, do not add leads that are already in urls_done (including the current url)
    urls_done = state.get('urls_done', {}).keys()
    state['urls_queue'].update({lead.url: lead.score for lead in answer.leads if lead.url not in urls_done})

    # sort urls_queue so the most likely urls are last in the dict
    state["urls_queue"] = dict(sorted(state["urls_queue"].items(), key=lambda item: item[1]))

    # The longer we search the more likely we already have the answer
    state['continue_searching_threshold'] += state['continue_searching_increment']
    return state

def continue_to_extract_data(state: AgentState):
    # check if we can stop searching
    if state.get('final_answer'):
        return END
    return "extract_data"

def get_step_message(node: str, state: AgentState):
    if node == 'extract_data':
        last_url = list(state['urls_done'].keys())[-1]
        return (f"Searched {last_url}, "
                f"continue searching score: {state['urls_done'][last_url]['continue_searching_score']}")
    return node
    # return f"Final answer found: {state['final_answer']}"

builder = StateGraph(AgentState)
builder.add_node("start_node", start_node)
builder.add_node("extract_data", extract_data)

builder.set_entry_point("start_node")
builder.add_edge("start_node", "extract_data")
builder.add_conditional_edges( # if answer is found go to END else go to check_url_for_leads
    "extract_data",
    continue_to_extract_data,
    {END: END, "extract_data": "extract_data"}
)

graph = builder.compile()

if __name__ == "__main__":
    for s in graph.stream(
            {"messages": [
                # HumanMessage(content="What llm models offered by Groq have a context window larger then 100k tokens on https://groq.com ?"),
                # HumanMessage(content="Does https://www.skybad.de sell Pipe Insulation ?"),
                HumanMessage(content="Does https://www.skybad.de sell Wilo Varios 15/1-8 pump if yes what is the price?"),
                # HumanMessage(content="Can u find the best price for 'Samsung Galaxy S24, 256GB' on https://tweakers.net/pricewatch/ ?"),
            ]}
        ):
        for node, state in s.items():
            print(get_step_message(node, state))
            if state.get('final_answer'):
                pprint(state['messages'])