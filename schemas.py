from typing import Optional, List

from pydantic import BaseModel, Field

class Lead(BaseModel):
    url: str = Field(
        description="The url of where information may be found, or to an url that might contain an url to the answer")
    score: int = Field(
        description="The score of the url, 0-100: Where 100 is the highest probability of this url containing the "
                    "answer, or bringing us closer to the answer.")

class Answer(BaseModel):
    generic_context_of_website: Optional[str] = Field(None,
        description="Looking at this single page can you describe in about 50 words what this website is about. If the "
                    "context_of_website is already given above, this field can be left empty, or if this page has "
                    "crucial info about the generic context that is not yet described in the current generic context, "
                    "you can make an improved version here, stay generic, also include the language of the website.")
    answer_candidate: str = Field(
        description="The complete or partial answer found on the page.")
    continue_searching_score: int = Field(
        description=("candidate answer to a question, can you reflect on the answers. What is the likelyhood that we "
                     "did not find a propper answer to the question because we did not look at enough subpages? Give a "
                     "score from 0 to 100 where 100 is that we deffently need to look at more webpages."))
    leads : List[Lead] = Field(
        description="The leads to urls that might contain the answer or parts of the answer, or bring us closer to the "
                    "final answer. if continue_searching_score is 0 or very low, you should still provide leads to "
                    "other pages that might contain the answer.")
    final_answer: Optional[str] = Field("Only fill this if you are sure this is the final answer that fully answers the"
                                        " question of the user, u can only fill this if continue_searching_score is "
                                        "smaller than te value provided above")

class Question(BaseModel):
  question: str = Field(
      description="The unanswered question the user has.")
  url: str = Field(
      description="The url for the website where the answer is expected. (Example: https://sub.tweakers.net)")

