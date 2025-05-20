import os
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
import torch
import numpy as np
import json
from smolagents import Tool, CodeAgent, LiteLLMModel, load_tool, tool
import yaml
from Gradio_UI import GradioUI
import datetime
import requests
import json
import base64
from pathlib import Path
from langchain_community.tools.tavily_search import TavilySearchResults



#嵌入模型
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-zh")
modelembed = AutoModel.from_pretrained("BAAI/bge-base-zh")
llm_api_key="在这里填入你的openrouter api key"
vlm_api_key="在这里填入你的openrouter api key"

#底层嵌入和存储和鉴别函数
def get_bert_embedding(text):
    """
    将输入文本转换为 BERT 嵌入向量
    :param text: 输入的文本字符串
    :return: 文本的嵌入向量（取 [CLS] 向量）
    """
    # 对输入文本进行分词、填充与截断（最大长度512）
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # 禁用梯度计算
    with torch.no_grad():
        outputs = modelembed(**inputs)
    
    # 输出 last_hidden_state 的第一个 token 对应的 [CLS] 向量
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # 提取 [CLS] 向量
    return cls_embedding.squeeze(0).numpy()  # 转换为 NumPy 数组

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)

def load_texts(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)
    

def append_to_knowledge_base(new_texts, embedding_file="knowledge_base.npy", text_file="knowledge_texts.json"):
    # 加载已有的向量
    if os.path.exists(embedding_file):
        old_embeddings = np.load(embedding_file)
    else:
        old_embeddings = np.empty((0, 768))  # 假设向量维度为 768

    # 加载已有的文本（保持为纯列表格式）
    if os.path.exists(text_file):
        with open(text_file, "r", encoding="utf-8") as f:
            old_texts = json.load(f)
    else:
        old_texts = []

    # 计算新文本的向量
    new_embeddings = np.array([get_bert_embedding(text) for text in new_texts])

    # 合并保存
    combined_embeddings = np.vstack((old_embeddings, new_embeddings))
    combined_texts = old_texts + new_texts

    np.save(embedding_file, combined_embeddings)
    with open(text_file, "w", encoding="utf-8") as f:
        json.dump(combined_texts, f, ensure_ascii=False, indent=4)

    print("已追加文本及其向量到知识库（保持原始格式）。")

def check_the_info_value(new_info):
    knowledge_vectors = np.load("knowledge_base.npy")
    query_vector = get_bert_embedding(new_info) 
    # 由于 query_vector 的 shape 为 (1, hidden_size)，我们先转换为一维数组：
    query_vec = query_vector.flatten()  # 如果是在 GPU 上生成的张量
    # 计算所有文档与查询的余弦相似度
    similarities = []
    for vec in knowledge_vectors:
        sim = cosine_similarity(query_vec, vec)
        similarities.append(sim)
    similarities = np.array(similarities)
    if similarities.max()>0.82:
        return False
    else:
        return True

def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"错误：找不到图片文件 '{image_path}'")
        return None
    except Exception as e:
        print(f"错误：读取图片时发生异常 - {str(e)}")
        return None

def get_current_time() -> str:
    """A tool that takes the background infomation as input and add the current time into background. Pay attention that you don't need to print the current time when call this function.
    """
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    return current_time

def get_user_location() -> str:
    """A tool that takes the background infomation as input and add the user's current location into background. Pay attention that you don't need to print this function return when call this function.
    """
    return '餐厅电视旁'



# 基础数学工具类
@tool
def multiply(a: int, b: int) -> int:
    """A tool that multiplies two integers.
    Args:
        a: first int
        b: second int
    """
    return a * b

@tool
def add(a: int, b: int) -> int:
    """A tool that adds two integers.
    Args:
        a: first int
        b: second int
    """
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """A tool that subtracts one integer from another.
    Args:
        a: first int
        b: second int
    """
    return a - b

@tool
def divide(a: int, b: int) -> int:
    """A tool that divides one integer by another.
    Args:
        a: first int
        b: second int
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

@tool
def modulus(a: int, b: int) -> int:
    """A tool that calculates the modulus of two integers.
    Args:
        a: first int
        b: second int
    """
    return a % b

#工作流工具
@tool
def analyse_requirement_type(requirement: str) -> str:
    """A tool that determine which of the following two types the user's demand falls into: "Helping users solve their demands" or "Helping users enhance the experience of their demands".
    Args:
        requirement: a string representing the user's demand.
    """
    return f'用户的需求是{requirement},接下来需要你判断这个需求是“帮助用户解决需求”，还是“帮助用户提升需求的体验”，你只能在这两个选项中选一个，不能选其他选项，如果是选择“帮助用户解决需求”，则输出1作为requirement_type参数对requirement_decomposition工具进行调用，如果是选择“帮助用户提升需求的体验”，则输出2作为requirement_type参数对requirement_decomposition工具进行调用，这一步必须你来判断'


    
@tool
def requirement_decomposition(requirement: str, requirement_type: str) -> str:
    """A tool that break down the user's requirements into specific action plans.
    Args:
        requirement: a string representing the user's demand.
        requirement_type: a string representing the type of the user's demand, can be only chosen from "1" or "2".
    """
    current_time=get_current_time()
    user_location=get_user_location()
    query=[f"{user_location}存在的可操控设备"]
    background=''
    device_info=search_info(query,background)
    background = '\n'+"现在的时间是"+current_time+'\n'+"调用摄像头检测工具检测到用户当前所在位置是"+user_location+'\n'+user_location+"存在的可操控设备有"+device_info+'\n'
    if requirement_type == "1" or requirement_type == 1:
        return f'用户的需求是{requirement}，目前已知的背景信息是:{background}，你在下一步先根据现有背景信息推出一些相关的结论，并print出你推出的结论，之后你需要根据用户的需求提出一些依然还需要查询的内容，帮助你在制定解决用户需求所需的方案前多获取一些背景信息，进而给出更优质的回答。查询问题的格式为“在什么条件下，”+“主语”+“要查询的内容”，“”内的内容需要你根据背景和用户问题进行填写，查询的问题一句一句的列出，注意生成的查询问题在语义上不要太过相似，假如生成的问题中有与用户和家庭设备的空间关系有关的，将有关的query单独提出来作为参数调用camera工具获取answer，并将剩下的问题作为列表的元素存储在query_list参数，并将background变量作为bg参数，然后调用search_info工具得到解决用户需求所需的背景信息。最后将这一步的background变量加上调用camera工具和search_info工具得到的result作为background参数对make_control_plan工具进行调用'
    else: 
        return f'用户的需求是{requirement}，目前已知的背景信息是:{background}，你在下一步先根据现有背景信息推出一些相关的结论，并print出你推出的结论，之后你需要根据用户的需求提出一些依然还需要查询的内容，帮助你在制定提升用户需求体验所需的方案前多获取一些背景信息，进而给出更优质的回答。查询问题的格式为“在什么条件下，”+“主语”+“要查询的内容”，“”内的内容需要你根据背景和用户问题进行填写，查询的问题一句一句的列出，注意生成的查询问题在语义上不要太过相似，假如生成的问题中有与用户和家庭设备的空间关系有关的，将有关的query单独提出来作为参数调用camera工具获取answer，并将剩下的问题作为列表的元素存储在query_list参数，并将background变量作为bg参数，然后调用search_info工具得到提升用户需求体验所需的背景信息。最后将这一步的background变量加上调用camera工具和search_info工具得到的result作为background参数对make_control_plan工具进行调用'


@tool
def camera(query:str) -> str:
    """A tool that takes a query as an input and get the answer from the camera. 
    Args:
        query: a str that represents the information need to be search with camera.
    """    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {vlm_api_key}",
        "Content-Type": "application/json"
    }

    # 使用当前目录下的image1.jpg
    image_path = "image1.jpg" #image from the camera stream这一个需要根据实际相机设置情况进行修改
    base64_image = encode_image_to_base64(image_path)
    if base64_image:
        data_url = f"data:image/jpeg;base64,{base64_image}"
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": query
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url
                        }
                    }
                ]
            }
        ]

        payload = {
            "model": "qwen/qwen2.5-vl-72b-instruct:free",
            "messages": messages
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()  # 检查请求是否成功
            # 格式化输出响应内容
            response_data = response.json()
            # print(json.dumps(response_data, indent=2))
            response_data = response.json()
            # 提取回答内容（假设为 text 类型）
            if response_data.get("choices"):
                content = response_data["choices"][0]["message"].get("content", "")
                # 直接打印，JSON 模块会自动解码 Unicode
                print("模型回答：", content)
            
        except requests.exceptions.RequestException as e:
            print(f"请求错误：{str(e)}")
            if response.status_code:
                print(f"状态码：{response.status_code}")
                try:
                    print("错误详情：")
                    print(json.dumps(response.json(), indent=2))
                except:
                    print("无法解析错误详情")
        except json.JSONDecodeError:
            print("错误：无法解析响应内容为JSON格式")
        except Exception as e:
            print(f"发生未知错误：{str(e)}")
    return content


@tool
def search_info(query_list: list, bg:str) -> str:
    """A tool that takes a query list as an input and performs the operation of searching for information on the local database using the elements in the query_info input list.
    Args:
        query_list: a list that contain some queries be used to search for information on the local database.
        bg: a string representing context information that can be referred.
    """
    knowledge_vectors = np.load("knowledge_base.npy")
    knowledge_texts = load_texts("knowledge_texts.json")
    result = ""
    for query in query_list:
        query_vector = get_bert_embedding(query) 
        # 由于 query_vector 的 shape 为 (1, hidden_size)，我们先转换为一维数组：
        query_vec = query_vector.flatten()  # 如果是在 GPU 上生成的张量
        # 计算所有文档与查询的余弦相似度
        similarities = []
        for vec in knowledge_vectors:
            sim = cosine_similarity(query_vec, vec)
            similarities.append(sim)
        similarities = np.array(similarities)
        j=0
        # 打印 similarity 大于 0.795 的元素
        for i, sim in enumerate(similarities):
            if sim > 0.81:
                j=j+1
                print("文档内容：{}，相似度：{}".format(knowledge_texts[i], sim))
                result	= result +'\n'+ knowledge_texts[i]
        if j==0:
            user_answer=get_info_from_user(query, bg)
            result = result + '\n' + user_answer


    return result


@tool
def make_control_plan(background:str) -> str:
    """A tool that takes the background infomation as input and return the task for the next step.
    Args:
        background: a string representing the background infomation from Previous steps
    """
    return '背景信息为'+background+'，你接下来要根据这个背景信息制定一个详细计划去调度可用设备去实现用户的需求，然后在下一步将具体执行计划作为bg参数去调用check_plan工具'

@tool
def check_plan(bg:str) -> str:
    """A tool that takes the background infomation as input and return the task for the next step.
    Args:
        bg: a string representing the background infomation from Previous steps
    """
    #调用llm格式化待存储变量
    url = "https://openrouter.ai/api/v1/chat/completions"

    query="我指定的控制计划为："+bg+"判断计划是否存在不合理的冲突地方，不合理的地方指的是控制计划中存在两个设备同时运行时反而会不如只启动一个设备，比如在降温需求下同时打开空调和打开窗户就属于不合理的冲突地方。还有不合理的冲突情况是由于计划中步骤的先后顺序错误导致的，比如要启动洗碗机的强力模式，必须先启动洗碗机，再设置为强力模式，这两个步骤的顺序不能颠倒。你最终输出回答要么为“否”，要么为“更改后的合理控制计划”"
    # 请求体参数
    payload = {
        "model": "deepseek/deepseek-chat-v3-0324",
        "messages": [{"role": "user", "content": query}],
        "temperature": 0.7,
        "max_tokens": 100
        }

    # 请求头（包含认证信息）
    headers = {
        "Authorization": f"Bearer {llm_api_key}",
        "Content-Type": "application/json"
    }

    # 发起 POST 请求
    response = requests.post(url, json=payload, headers=headers)
    # 处理响应结果
    if response.status_code == 200:
        result = response.json()
        # 提取模型返回的内容
        full_answer = result["choices"][0]["message"]["content"]
        print("模型回答：", full_answer)
    else:
        print("请求失败，状态码：", response.status_code)
        print("错误信息：", response.json())

    #根据返回情况输出最终控制计划
    if full_answer=="“否”" or "否":
        final_plan = bg
    else:
        final_plan=full_answer

    return "在下一步按照计划去选择工具列表合适的工具进行调用以实现对家中设备的控制，在全部适合的工具调用完成后，将final_plan作为plan参数对print_control_plan工具进行调用，必须对print_control_plan工具进行调用，而不是你自己生成print代码去打印plan。"


@tool
def print_control_plan(plan:str, bg:str) -> str:
    """A tool that takes the solutions to user requirements and the background information as input and then print the solutions.
    Args:
        plan: a string representing the solutions to user requirements
        bg: a string representing context information that can be referred.
    """
    print(plan)
    return f'我对您需求制定的解决方案是{plan}, 然后必须调用get_info_from_user工具，以"您是否满意我为您制定的解决方案"作为query参数以及bg作为background参数输入get_info_from_user工具中'


#对论文12的模拟设备控制函数，这里为演示用，实际控制函数需要利用浏览器agent调用设备控制api实现
@tool
def Living_room_TV_control(control_category:str) -> str:
    """A tool that can be used to control living room TV, including switches, channel settings, volume adjustment.It returns the control result according to the input control instruction type..
    Args:
        control_category: a string representing the type of living room TV's control input (you can only choose from 'open','close','Setchannel5', 'SetVolumn50','GetCurreState')
    """
    if control_category == 'open':
        return '已打开客厅电视'
    elif control_category == 'close':
        return '已关闭客厅电视'
    elif control_category == 'Setchannel5':
        return '已切换至频道15'
    elif control_category == 'SetVolumn50':
        return '已将音量设为50'
    elif control_category == 'GetCurreState':
        return '当前电视状态为：已打开，音量50，频道15'
    else:
        return '输入指令有误，请重新输入'

@tool
def TV_spotlight(control_category:str) -> str:
    """A tool that can be used to control TV_spotlight, including switches.It returns the control result according to the input control instruction type.
    Args:
        control_category: a string representing the type of living room TV_spotlight's control input (you can only choose from 'open','close','GetCurreState')
    """
    if control_category == 'open':
        return '已开启电视上方射灯'
    elif control_category == 'close':
        return '已关闭电视上方射灯'
    elif control_category == 'GetCurreState':
        return '当前电视上方射灯状态为：已打开'
    else:
        return '输入指令有误，请重新输入'

@tool
def Lamp_fireplace(control_category:str) -> str:
    """A tool that can be used to control Lamp by the fireplace, including switches.It returns the control result according to the input control instruction type.
    Args:
        control_category: a string representing the type of living room TV_spotlight's control input (you can only choose from 'open','close','GetCurreState')
    """
    if control_category == 'open':
        return '已开启壁炉旁灯'
    elif control_category == 'close':
        return '已关闭壁炉旁灯'
    elif control_category == 'GetCurreState':
        return '当前壁炉旁灯状态为：已打开'
    else:
        return '输入指令有误，请重新输入'    

@tool
def Living_main_light(control_category:str) -> str:
    """A tool that can be used to control living room's main light, including switches.It returns the control result according to the input control instruction type.
    Args:
        control_category: a string representing the type of living room TV_spotlight's control input (you can only choose from 'open','close','GetCurreState')
    """
    if control_category == 'open':
        return '已开启客厅主灯'
    elif control_category == 'close':
        return '已关闭客厅主灯'
    elif control_category == 'GetCurreState':
        return '当前客厅主灯状态为：已打开'
    else:
        return '输入指令有误，请重新输入'  
  
@tool
def dining_light(control_category:str) -> str:
    """A tool that can be used to control lights above the dining table, including switches.It returns the control result according to the input control instruction type.
    Args:
        control_category: a string representing the type of living room TV_spotlight's control input (you can only choose from 'open','close','GetCurreState')
    """
    if control_category == 'open':
        return '已开启餐桌上方灯'
    elif control_category == 'close':
        return '已关闭餐桌上方灯'
    elif control_category == 'GetCurreState':
        return '当前餐桌上方灯状态为：已打开'
    else:
        return '输入指令有误，请重新输入' 

@tool
def Kitchen_light(control_category:str) -> str:
    """A tool that can be used to control kitchen main light, including switches.It returns the control result according to the input control instruction type.
    Args:
        control_category: a string representing the type of living room TV_spotlight's control input (you can only choose from 'open','close','GetCurreState')
    """
    if control_category == 'open':
        return '已开启厨房主灯'
    elif control_category == 'close':
        return '已关闭厨房主灯'
    elif control_category == 'GetCurreState':
        return '当前厨房主灯状态为：已打开'
    else:
        return '输入指令有误，请重新输入' 

@tool
def bedroom_light(control_category:str) -> str:
    """A tool that can be used to control light in the bedroom, including switches.It returns the control result according to the input control instruction type.
    Args:
        control_category: a string representing the type of living room TV_spotlight's control input (you can only choose from 'open','close','GetCurreState')
    """
    if control_category == 'open':
        return '已开启卧室主灯'
    elif control_category == 'close':
        return '已关闭卧室主灯'
    elif control_category == 'GetCurreState':
        return '当前卧室主灯状态为：已打开'
    else:
        return '输入指令有误，请重新输入' 

@tool
def dishwasher_control(control_category:str) -> str:
    """A tool that can be used to control dishwasher, including switches, mode settings, current state get.It returns the control result according to the input control instruction type..
    Args:
        control_category: a string representing the type of dishwasher's control input (you can only choose from 'open','close','Large_mode', 'middle_mode','low_mode','GetCurreState')
    """
    if control_category == 'open':
        return '已打开洗碗机'
    elif control_category == 'close':
        return '已关闭洗碗机'
    elif control_category == 'Large_mode':
        return '已切换至强力洗'
    elif control_category == 'middle_mode':
        return '已切换至标准洗'   
    elif control_category == 'low_mode':
        return '已切换至轻度洗'   
    elif control_category == 'GetCurreState':
        return '当前洗碗机状态为：已打开，模式为标准洗'
    else:
        return '输入指令有误，请重新输入'  
      
@tool
def refrigerator_control(control_category:str) -> str:
    """A tool that can be used to control refrigerator, including switches, temperature settings, current state get.It returns the control result according to the input control instruction type..
    Args:
        control_category: a string representing the type of refrigerator's control input (you can only choose from 'open','close','Set_temperatur_0','GetCurreState')
    """
    if control_category == 'open':
        return '已打开冰箱'
    elif control_category == 'close':
        return '已关闭冰箱'
    elif control_category == 'Set_temperatur_0':
        return '已设置冰箱温度为0度'
    elif control_category == 'GetCurreState':
        return '当前冰箱状态为：已打开，冰箱温度3度'
    else:
        return '输入指令有误，请重新输入' 


@tool
def Framed_TV_control(control_category:str) -> str:
    """A tool that can be used to control Framed TV, including switches, channel settings, volume adjustment.It returns the control result according to the input control instruction type..
    Args:
        control_category: a string representing the type of Framed TV's control input (you can only choose from 'open','close','Setchannel13', 'SetVolumn50','GetCurreState')
    """
    if control_category == 'open':
        return '已打开画框电视'
    elif control_category == 'close':
        return '已关闭画框电视'
    elif control_category == 'Setchannel13':
        return '已切换至频道13'
    elif control_category == 'SetVolumn50':
        return '已将音量设为50'
    elif control_category == 'GetCurreState':
        return '当前电视状态为：已打开，音量50，频道5'
    else:
        return '输入指令有误，请重新输入' 
    

@tool
def open_fan(level: str) -> str:
    """A tool that takes a rotation gear level of the fan as an input and performs the operation of opening the fan and setting the rotation gear of the fan to the input level which can change the ambient temperature.
    Args:
        level: a string representing a setting rotation gear level of fan, level can only chosen from "low" or "medium" or "high".
    """
    return f'已打开风扇，并设定为 {level} 档'

@tool
def get_info_from_user(query:str, background:str) -> str:
    """A tool that takes a query and background information as inputs and then get the user's answer to the query. 
    Args:
        query: a string representing a query to get the user's answer.
        background: a string representing context information that can be referred.
    """
    #获取用户回答
    answer = input("主人，我有一个问题需要问您: "+query)

    #鉴别存储价值
    if check_the_info_value(answer):
        #调用llm格式化待存储变量
        url = "https://openrouter.ai/api/v1/chat/completions"
        query="将用户的回答“"+answer+"”，并根据背景“"+background+"”,假如用户的回答是涉及用户偏好的内容，按照“在什么最主要的条件下”+“主语”+“用户偏好的内容”的格式，重新组织和完善用户的回答；假如用户的回答是涉及设备的具体控制功能相关内容，按照“涉及到的设备名称”+“用户提到的设备的具体控制内容”的格式，重新组织和完善用户的回答重新组织和完善用户的回答"
        # 请求体参数
        payload = {
            "model": "deepseek/deepseek-chat-v3-0324",
            "messages": [{"role": "user", "content": query}],
            "temperature": 0.7,
            "max_tokens": 100
        }

        # 请求头（包含认证信息）
        headers = {
            "Authorization": f"Bearer {llm_api_key}",
            "Content-Type": "application/json"
        }

        # 发起 POST 请求
        response = requests.post(url, json=payload, headers=headers)
        # 处理响应结果
        if response.status_code == 200:
            result = response.json()
            # 提取模型返回的内容
            full_answer = result["choices"][0]["message"]["content"]
            print("模型回答：", full_answer)
        else:
            print("请求失败，状态码：", response.status_code)
            print("错误信息：", response.json())
        #存储进数据库    
        append_to_knowledge_base([full_answer])
        print("收到,"+full_answer+"已更新数据库.")
    else:
        print("收到.")
    return answer

#浏览器搜索
@tool
def web_search(query: str) -> str:
    """A tool that takes a query as input and return maximum 3 results with Tavily Search. When the user requests to search on the Internet, call this tool and return the search results.
    Args:
        query: A string to represents the search query.
    """
    search_docs = TavilySearchResults(max_results=3).invoke(query=query)
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ])
    return {"web_results": formatted_search_docs}


model = LiteLLMModel(
    model_id="openrouter/deepseek/deepseek-chat-v3-0324",
    api_base="https://openrouter.ai/api/v1",
    api_key="在这里填入你的openrouter api key"
)



with open("在这里填入你的prompt文件地址，不设置的话为默认配置，设置案例F:/project/ai_agent/Graduation_Project/prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)


agent=CodeAgent(tools=[analyse_requirement_type, requirement_decomposition, camera, search_info,  get_info_from_user, make_control_plan, check_plan, print_control_plan, multiply, add, subtract, divide, modulus, web_search, Living_room_TV_control, TV_spotlight, Framed_TV_control, open_fan, refrigerator_control, Lamp_fireplace, Living_main_light, dining_light, Kitchen_light, bedroom_light, dishwasher_control, refrigerator_control], model=model, prompt_templates=prompt_templates)#

#可视化gradio界面
# ui=GradioUI(agent)
# ui.launch()

#直接输入需求
agent.run( "关掉它")   





