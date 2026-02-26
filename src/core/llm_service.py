# src/core/llm_service.py
import os
from openai import OpenAI
import httpx

class QwenLLMService:
    def __init__(self, model="qwen-plus"):
        """
        通用的 Qwen 大模型服务类
        默认使用 qwen-plus (性价比高，适合文本翻译和处理)
        """
        self.api_key = os.environ.get("DASHSCOPE_API_KEY")
        self.model = model
        
        if not self.api_key:
            raise ValueError("未找到 DASHSCOPE_API_KEY 环境变量，请先设置！")

        # 强制阿里云域名不走系统代理
        os.environ["NO_PROXY"] = "dashscope.aliyuncs.com,aliyuncs.com"

        # 配置底层的 http 客户端（设置 60 秒超时）
        http_client = httpx.Client(timeout=httpx.Timeout(60.0))

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", 
            http_client=http_client
        )

    def chat(self, user_prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        """
        通用的文本对话接口，未来可以做任何文本处理任务
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1, # 翻译/提取任务需要低温度，保证确定性
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ LLM 请求抛出异常: {e}")
            raise e

    def translate_instructions(self, instructions: list) -> list:
        """
        [具体业务方法] 将中文任务指令翻译为专业的机器人英文指令
        """
        if not instructions:
            return []
            
        system_prompt = """
        You are an expert robotic data annotator and professional translator. 
        Your task is to translate the following robotic task instructions from Chinese to English.
        
        RULES:
        1. Maintain the exact original meaning.
        2. Use professional robotics terminology (e.g., 'gripper', 'robotic arm', 'align', 'pick', 'place', 'insert').
        3. Keep the translation concise and action-oriented.
        4. RETURN ONLY THE TRANSLATED SENTENCES, separated by newlines. 
        5. DO NOT add any extra markdown, numbering, quotes, or explanations.
        """
        
        # 将列表拼成多行字符串交给模型
        user_prompt = "\n".join(instructions)
        
        result_text = self.chat(user_prompt, system_prompt)
        
        # 将返回的多行文本重新切分成列表
        translated_list = [line.strip() for line in result_text.split('\n') if line.strip()]
        return translated_list
    
    def translate_task_name(self, chinese_text: str) -> str:
        """
        将中文动作描述翻译为标准的英文下划线格式 (verb_noun)
        示例：输入 "拿苹果" -> 输出 "pick_apple"
        """
        if not chinese_text.strip():
            return ""

        system_prompt = """
        You are a professional robotics dataset namer.
        Translate the user's Chinese robot action into a concise English phrase.
        
        RULES:
        1. Use ONLY lowercase letters and underscores (_).
        2. Format: verb_object (e.g., 'pick_apple', 'open_drawer').
        3. Use professional robotics verbs (pick, place, push, pull, open, close, insert).
        4. RETURN ONLY THE PHRASE, no explanation, no quotes.
        """
        
        # 调用通用的 chat 接口
        result = self.chat(chinese_text, system_prompt)
        # 简单清理：转小写，空格换下划线，过滤非法字符
        cleaned = result.strip().lower().replace(" ", "_")
        return cleaned