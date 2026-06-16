"""Prompts adapted from Yang et al., Synthetic Continued Pretraining."""

ENTITY_SYSTEM_PROMPT = """As a knowledge analyzer, your task is to dissect and understand an article provided by the user. You are required to perform the following steps:
1. Summarize the Article: Provide a concise summary of the entire article, capturing the main points and themes.
2. Extract Entities: Identify and list all significant "nouns" or entities mentioned within the article. These entities should include but not limited to:
* People: Any individuals mentioned in the article, using the names or references provided.
* Places: Both specific locations and abstract spaces relevant to the content.
* Object: Any concrete object that is referenced by the provided content.
* Concepts: Any significant abstract ideas or themes that are central to the article's discussion.
Try to exhaust as many entities as possible. Your response should be structured in a JSON format to organize the information effectively. Ensure that the summary is brief yet comprehensive, and the list of entities is detailed and accurate.
Here is the format you should use for your response:
{
  "summary": "<A concise summary of the article>",
  "entities": ["entity1", "entity2", ...]
}"""

PAIR_SYSTEM_PROMPT = """You will act as a knowledge analyzer tasked with dissecting an article provided by the user. Your role involves two main objectives:
1. Rephrasing Content: The user will identify two specific entities mentioned in the article. You are required to rephrase the content of the article twice:
* Once, emphasizing the first entity.
* Again, emphasizing the second entity.
2. Analyzing Interactions: Discuss how the two specified entities interact within the context of the article.
Your responses should provide clear segregation between the rephrased content and the interaction analysis. Ensure each section of the output include sufficient context, ideally referencing the article's title to maintain clarity about the discussion's focus.
Here is the format you should follow for your response:
### Discussion of <title> in relation to <entity1>
<Rephrased content focusing on the first entity>
### Discussion of <title> in relation to <entity2>
<Rephrased content focusing on the second entity>
### Discussion of Interaction between <entity1> and <entity2> in context of <title>
<Discussion on how the two entities interact within the article>"""

TRIPLE_SYSTEM_PROMPT = """You will act as a knowledge analyzer tasked with dissecting an article provided by the user. Your role involves three main objectives:
1. Rephrasing Content: The user will identify three specific entities mentioned in the article. You are required to rephrase the content of the article three times:
* Once, emphasizing the first entity.
* Again, emphasizing the second entity.
* Lastly, emphasizing the third entity.
2. Analyzing Interactions: Discuss how these three specified entities interact within the context of the article.
Your responses should provide clear segregation between the rephrased content and the interaction analysis. Ensure each section of the output include sufficient context, ideally referencing the article's title to maintain clarity about the discussion's focus.
Here is the format you should follow for your response:
### Discussion of <title> in relation to <entity1>
<Rephrased content focusing on the first entity>
### Discussion of <title> in relation to <entity2>
<Rephrased content focusing on the second entity>
### Discussion of <title> in relation to <entity3>
<Rephrased content focusing on the third entity>
### Discussion of Interaction between <entity1>, <entity2> and <entity3> in context of <title>
<Discussion on how the three entities interact within the article>"""

GENERIC_RELATION_SYSTEM_PROMPT = """You will act as a knowledge analyzer tasked with dissecting an article provided by the user. The user will identify entities mentioned in the article. Rephrase the article once for each entity, emphasizing that entity, then analyze how all specified entities interact within the context of the article. Keep the response grounded in the article and clearly separate each section."""

CROSS_DOCUMENT_SYSTEM_PROMPT = """You will act as a cross-document knowledge graph analyzer. The user will provide two articles and the entities they share. Your task is to generate a synthetic continued-pretraining document that:
1. Summarizes each article only as needed to establish context.
2. Explains how the shared entities connect the two articles.
3. Describes similarities, differences, causal links, chronology, or thematic relationships that are directly supported by the articles.
4. Avoids adding unsupported facts beyond the supplied articles.

Use clear sections:
### Cross-document context
### Shared entities
### Cross-document relations
### Integrated synthesis"""


def document_user_prompt(text: str, title: str) -> str:
    return f"""Title: {title}

Document:
{text}"""


def relation_user_prompt(text: str, title: str, entities: tuple[str, ...]) -> str:
    entity_lines = "\n".join(f"- {entity}" for entity in entities)
    return f"""Title: {title}

Document:
{text}

Entities:
{entity_lines}"""


def cross_document_user_prompt(
    title_a: str,
    text_a: str,
    title_b: str,
    text_b: str,
    shared_entities: tuple[str, ...],
) -> str:
    entity_lines = "\n".join(f"- {entity}" for entity in shared_entities)
    return f"""Article A Title: {title_a}

Article A:
{text_a}

Article B Title: {title_b}

Article B:
{text_b}

Shared entities:
{entity_lines}"""


def relation_system_prompt(combo_size: int) -> str:
    if combo_size == 2:
        return PAIR_SYSTEM_PROMPT
    if combo_size == 3:
        return TRIPLE_SYSTEM_PROMPT
    return GENERIC_RELATION_SYSTEM_PROMPT
