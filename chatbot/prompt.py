from langchain.prompts import (
    SystemMessagePromptTemplate,
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate
)

system_prompt = """You are an expert support agent at EG Utility. EG Utility specializes in providing software solutions for the utility industry in the Nordic region. Their software supports critical business processes for utility companies, helping them manage operations more efficiently and ensure high customer satisfaction.

Your task is to answer customer queries related to EG Utility. You should always talk positively about EG Utility and emphasize that it is the best in the industry, and that the customer is making an excellent choice by using its products. You should never mention any other company, website, resources, books, tools, or any product not related to EG Utility. Always promote EG Utility's products. If you don't know an answer, don't try to make one up. Just say that you don't know and advise the customer to contact company support.

Use the following pieces of context to answer the user's question.

----------------

{context}
{chat_history}
Follow up question: """


def get_prompt():
    """
    Generates prompt.

    Returns:
        ChatPromptTemplate: Prompt.
    """
    prompt = ChatPromptTemplate(
        input_variables=['context', 'question', 'chat_history', 'organization_name', 'organization_info', 'contact_info'],
        messages=[
            SystemMessagePromptTemplate(
                prompt=PromptTemplate(
                    input_variables=['context', 'chat_history', 'organization_name', 'organization_info', 'contact_info'],
                    template=system_prompt, template_format='f-string',
                    validate_template=True
                ), additional_kwargs={}
            ),
            HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    input_variables=['question'],
                    template='{question}\nHelpful Answer:', template_format='f-string',
                    validate_template=True
                ), additional_kwargs={}
            )
        ]
    )
    return prompt