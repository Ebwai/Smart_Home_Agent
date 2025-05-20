from transformers import  AutoTokenizer, AutoModel
import torch
import numpy as np
import json

#模型和分词器的加载
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-zh")
model = AutoModel.from_pretrained("BAAI/bge-base-zh")

#文本向量化工具
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # 输出 last_hidden_state 的第一个 token 对应的 [CLS] 向量
    cls_embedding = outputs.last_hidden_state[:, 0, :]  
    return cls_embedding.squeeze(0).numpy()  

#数据库构建内容案例
# text_fragments = [
#     "用户家庭有一个客厅，一个卧室，一个厨房，一个卫生间，一个书房，一个阳台",
#     "家庭中的全部可控制设备：客厅电视，画框电视，餐厅主灯，餐厅射灯，壁炉，空调，窗帘，窗户，电扇，音响",
#     "客厅电视：位置在客厅的植物旁。",
#     "客厅电视：电视的可操作内容：开关，音量[0-100%]，频道[频道1-10都可以切换],获取当前电视的信息",
#     "画框电视：位置在客厅餐柜旁。",
#     "电视的可操作内容：开关，音量[0-100%]，频道[频道1-10都可以切换]，获取当前电视的信息",
#     "餐厅主灯：位置在餐厅的餐桌的正上方。",
#     "餐厅主灯：可操作内容：开关，亮度[0-100%]，颜色[暖调，暗调]，获取当前灯的状态",
#     "餐厅射灯：位置在餐厅的餐桌的靠墙端墙壁上。",
#     "餐厅射灯：可操作内容：开关",
#     "壁炉：位置在客厅的沙发旁。"
#     "频道5是体育频道，频道6是新闻频道，频道7是电影频道，频道8是电视剧频道，频道9是儿童频道，频道10是音乐频道",
#     "用户经常看频道9，因为该频道经常播放有趣的内容",
#     "用户喜好的室内温度是22摄氏度",
#     "房间里可控制设备有窗户，空调，电扇，电视机，音响，窗帘",
#     "用户在看书时喜欢暖色的光",
#     "用户一般在晚上22点睡觉",
#     "用户在室内温度不超过30摄氏度时喜好通过开窗降低室内温度"
# ]

text_fragments = [
    "用户家庭有一个客厅，一个卧室，一个厨房，一个卫生间，一个书房，一个阳台",
    "家庭中的全部可控制设备：客厅电视，画框电视，餐厅主灯，餐厅射灯，壁炉，空调，窗帘，窗户，电扇，音响",
    "客厅电视：位置在客厅的植物旁。",
    "客厅电视：电视的可操作内容：开关，音量[0-100%]，频道[频道1-10都可以切换],获取当前电视的信息。",
    "客厅电视：可使用Living_room_TV_control工具进行控制",
    "画框电视：位置在客厅餐柜旁。",
    "画框电视：电视的可操作内容：开关，音量[0-100%]，频道[频道1-10都可以切换]，获取当前电视的信息",
    "画框电视：可使用Framed_TV_control工具进行控制",
    "餐厅主灯：位置在餐厅的餐桌的正上方。",
    "餐厅主灯：可操作内容：开关，亮度[0-100%]，颜色[暖调，暗调]，获取当前灯的状态",
    "餐厅射灯：位置在餐厅的餐桌的靠墙端墙壁上。",
    "餐厅射灯：可操作内容：开关",
    "电视机射灯：位置在客厅的电视机上方",
    "电视机射灯：可操作内容：开关，获取当前灯的状态。",
    "电视机射灯：可使用TV_spotlight工具进行控制",
    "壁炉：位置在客厅的沙发旁。"
    "频道5是体育频道，频道13是新闻频道，频道7是电影频道，频道8是电视剧频道，频道9是儿童频道，频道10是音乐频道",
    "用户经常看频道9，因为该频道经常播放有趣的内容",
    "用户喜好的室内温度是22摄氏度",
    "用户在看书时喜欢暖色的光",
    "用户一般在晚上22点睡觉",
    "用户在室内温度不超过30摄氏度时喜好通过开窗降低室内温度"
]

# 计算所有文本片段的向量
embeddings = np.array([get_bert_embedding(text) for text in text_fragments])


# 存储向量数据到 knowledge_base.npy
np.save("knowledge_base.npy", embeddings)
print("向量已存储到 knowledge_base.npy")

# 6. 存储文本片段到 JSON 文件（方便后续检索）
with open("knowledge_texts.json", "w", encoding="utf-8") as f:
    json.dump(text_fragments, f, ensure_ascii=False, indent=4)
print("文本片段已存储到 knowledge_texts.json")
