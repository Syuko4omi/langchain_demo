import openai
import os
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import SimpleSequentialChain
import templates

openai.api_key = os.environ["OPENAI_API_KEY"]


def create_chain(template):
    llm = ChatOpenAI()
    template = template
    prompt = HumanMessagePromptTemplate.from_template(template)
    prompt = ChatPromptTemplate.from_messages([prompt])
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain


def proofreader(sentence, style):
    style_trans_template = (
        templates.keitai_to_jyoutai_template
        if style == "常体"
        else templates.jyoutai_to_keitai_template
    )
    proofread_chain = create_chain(templates.proofread_template)
    style_trans_chain = create_chain(style_trans_template)
    proofread_and_style_trans_chain = SimpleSequentialChain(
        chains=[proofread_chain, style_trans_chain], verbose=True
    )
    return proofread_and_style_trans_chain.run(sentence)


if __name__ == "__main__":
    original_sentence = "ダイエットのためには食事管理が重要と言われていますが、バランスを考えずにリンゴだけを食べる方法ではリバウンドを招き、かえって以前よりも体重が増えてしまうことになり、痩せにくい体質にもなってしまいます。"
    original_style = "敬体"
    improved_text = proofreader(original_sentence, original_style)
    print(improved_text)
