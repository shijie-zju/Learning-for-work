import os
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
#from langchain_chroma import Chroma
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


load_dotenv()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def retrieve(retriever, question='solvent'):
    documents = retriever.invoke(question)
    return format_docs(documents)

def generate(prompt, llm, context, question):
    rag_chain = prompt | llm | StrOutputParser()
    rag_chain = prompt
    generation = rag_chain.invoke({
        'context': context,
        'question': question
    })
    print(f'generation{generation}')
    return generation.replace('\n', '').replace('\\', '')

def get_all_file(dir_path, suffix='.pdf'):
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"{dir_path} is not a directory")

    return [file for file in dir_path.iterdir() if file.is_file() and file.suffix == suffix]


# save output dict to json file
def save_dict_as_json(dictionary, filename):
    # 确保字典的值也是合法的JSON
    # 这里假设字典的值已经是合法的JSON字符串
    print(f'Save to {filename}')
    with open(filename, 'w', encoding='utf-8') as f:
        # 将整个字典序列化为JSON字符串
        json.dump(dictionary, f, ensure_ascii=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Retrieval")
    parser.add_argument('--pdf_folder', default='paper')
    parser.add_argument('--model', default='llama3.1:70b-instruct-fp16')
    args = parser.parse_args()

    # model
    MODEL = args.model
    #model = Ollama(model=MODEL)
    model = ChatOpenAI(temperature=0.1)
    embeddings_model = OllamaEmbeddings(model=MODEL)
    print(f"Using model: {MODEL}")

    # prompt
    template = """
    Task
    Extract information about solvent molecules from the provided chemical paper context and structure it in JSON format. For each solvent molecule, include the following fields:

    `Example
    Context:
    "Electrolyte solutions were prepared using a mixture of propylene carbonate (PC), dimethyl carbonate (DMC), and 1,3-dioxolane:1,2-dimethoxyethane (DME). The solvents PC and DMC were mixed in a 1:1 volume ratio."
    
    Question:
    What is the solvent molecule used in the context?
    
    Output:
    [
        {{
            "common_name": "propylene carbonate",
            "systematic_name": null,
            "chemical_formula": "C4H6O3"
        }},
        {{
            "common_name": "1,2-dimethoxyethane",
            "systematic_name": "1,2-dimethoxyethane",
            "chemical_formula": "C4H10O2"
        }},
        ...
    ]`
    
    Context:
    {context}
    Question:
    {question}
    
    Note: Don't output extra content
    """
    prompt = PromptTemplate.from_template(template)

    output = dict()
    for file in get_all_file(args.pdf_folder):
        print(f"Processing {file}")

        file_name = file.stem
        vectorstore = Chroma(persist_directory=f'chroma_data/{file_name}', embedding_function=embeddings_model)
        _retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        context = retrieve(_retriever, question='solvent')
        context = 'hey, siri'
        response = generate(prompt, model, context, 'What is the solvent molecule used in the context? Extract their full names and return them in the format shown in example')
        print(str(file) + '\n' +response)
        output[file_name] = response

    # save
    save_dict_as_json(output, 'output.json')

