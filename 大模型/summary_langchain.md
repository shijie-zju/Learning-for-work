# Langchain
LangChain是一个用于开发由大型语言模型(llm)支持的应用程序的框架。<br/>
LangChain简化了LLM应用程序生命周期的每个阶段:<br/>
开发:使用LangChain的开源构建块、组件和第三方集成来构建应用程序 / 使用LangGraph构建具有一流流和人在循环支持的有状态代理。<br/>
产品化:使用LangSmith来检查、监控和评估您的链，以便您可以自信地持续优化和部署。<br/>
部署:使用LangGraph Cloud将您的LangGraph应用程序转变为生产就绪的api和助手。<br/>

0.准备工作
```commandline
!pip install langchain
import os
api_key = os.getenv("OPENAI_API_KEY") #环境变量中获取key
```
1.普通调用
```commandline
#1首先导入LLM包装器，然后即可调用大语言模型
from langchain.llms import OpenAI

#初始化一个智能体名为llm
llm = OpenAI(temperature=0.9) #设置温度增加随机性
text = "What would be a good company name for a company that makes colorful socks?"
print(llm(text))
```
2.提示模板promptTemplate
```commandline
#2导入提示模版的功能
from langchain.prompts import PromptTemplate

#写一个提示模版
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

#我们可以调用. format 方法来格式化它。
print(prompt.format(product="colorful socks"))
```
3.链：在多步骤工作流中组合LLM和提示
```commandline
#3导入链的功能
from langchain.chains import LLMChain

chain = LLMChain(llm=llm, prompt=prompt)

#现在我们可以运行该链，只指定产品
chain.run("colorful socks")
```
4.智能体Agent:基于用户输入的动态调用链<br/>
工具（tools): 执行特定任务的功能。这可以是: Google 搜索、数据库查找、 Python REPL、其他链。工具的接口目前是一个函数，预计将有一个字符串作为输入，一个字符串作为输出。<br/>
大语言模型（LLM）: 为代理提供动力的语言模型。<br/>
代理（agents）: 要使用的代理。这应该是引用支持代理类的字符串。因为本教程主要关注最简单、最高级别的 API，所以它只涉及使用标准支持的代理。<br/>
```commandline
#4导入智能体中的调用工具和智能体功能
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
#导入语言模型
from langchain.llms import OpenAI
 
#首先设置语言模型
llm = OpenAI(temperature=0)
 
#接下来加载一些工具，其中llm-math工具需要用到LLM，也得放进去
tools = load_tools(["serpapi", "llm-math"], llm=llm) #serpapi也需要api才能运行
#tools = load_tools(["llm-math"], llm=llm)
#最后初始化含有工具的智能体，以及我们想用要的智能体类型：零样本学习的反应式的描述型智能助手
#verbose：布尔值参数。如果为True，则代理在执行任务时将提供详细的反馈和说明。如果为False，代理可能只会返回在执行时产生的最终结果。
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
 
# Now let's test it out!
agent.run("What was the high temperature in SF yesterday in Fahrenheit? What is that number raised to the .023 power?")
```













