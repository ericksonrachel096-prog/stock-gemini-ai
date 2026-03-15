import os
import hashlib
import xmltodict
import requests
import json
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from google import genai
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# 加载环境变量（本地测试用，Zeabur通过平台环境变量注入）
load_dotenv()
app = FastAPI(title="股票基金预测Gemini AI助手")
scheduler = AsyncIOScheduler(timezone=os.getenv("TZ", "Asia/Shanghai"))

# -------------------------- 核心配置初始化 --------------------------
# Gemini API初始化
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL = "gemini-2.0-flash"  # 日常用flash，深度分析可换gemini-2.5-pro

# 股票数据源配置（Alpha Vantage）
STOCK_API_KEY = os.getenv("STOCK_API_KEY")
STOCK_API_URL = "https://www.alphavantage.co/query"

# 微信配置
WECHAT_TOKEN = os.getenv("WECHAT_TOKEN")

# -------------------------- 股票数据获取与预测核心逻辑 --------------------------
def get_stock_data(stock_code: str):
    """获取股票/基金实时行情与历史数据"""
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": stock_code,
        "apikey": STOCK_API_KEY,
        "outputsize": "compact"
    }
    try:
        response = requests.get(STOCK_API_URL, params=params, timeout=10)
        if response.status_code != 200:
            return None
        data = response.json()
        # 过滤无效数据
        if "Time Series (Daily)" not in data:
            return None
        return data
    except Exception as e:
        print(f"获取行情数据失败：{e}")
        return None

def gemini_stock_analysis(stock_code: str, stock_data: dict):
    """调用Gemini生成股票/基金预测分析报告"""
    # 合规Prompt模板，可按需优化
    prompt = f"""
    角色：你是专业合规的金融市场AI分析助手，仅基于提供的行情数据给出客观分析参考，不构成任何投资建议。
    标的代码：{stock_code}
    行情数据：{stock_data}

    输出要求：
    1. 核心趋势预判（短期/中期），附带概率参考
    2. 关键支撑位与压力位参考
    3. 核心风险点提示
    4. 结尾必须添加固定免责声明：投资有风险，入市需谨慎，本内容仅为AI技术分析参考，不构成任何投资操作建议
    5. 语言通俗易懂，不夸大收益，不做保本承诺
    """
    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        return response.text
    except Exception as e:
        print(f"Gemini调用失败：{e}")
        return "AI分析服务暂时不可用，请稍后重试"

# -------------------------- 微信公众号接入接口 --------------------------
@app.get("/wechat")
async def wechat_verify(
    signature: str = Query(...),
    timestamp: str = Query(...),
    nonce: str = Query(...),
    echostr: str = Query(...)
):
    """微信服务器校验接口"""
    # 微信签名校验
    tmp_list = sorted([WECHAT_TOKEN, timestamp, nonce])
    tmp_str = "".join(tmp_list).encode("utf-8")
    tmp_sign = hashlib.sha1(tmp_str).hexdigest()
    if tmp_sign == signature:
        return int(echostr)
    raise HTTPException(status_code=403, detail="微信校验失败")

@app.post("/wechat")
async def wechat_message(request: Request):
    """微信用户消息处理接口"""
    xml_data = await request.body()
    msg_dict = xmltodict.parse(xml_data)["xml"]
    from_user = msg_dict["FromUserName"]
    to_user = msg_dict["ToUserName"]
    content = msg_dict.get("Content", "").strip()

    # 默认回复
    reply_content = "欢迎使用股票基金预测AI助手\n发送格式：股票/基金代码+预测\n例：000001 预测"

    # 处理用户预测请求
    if "预测" in content and len(content.split()) >= 1:
        stock_code = content.split()[0]
        stock_data = get_stock_data(stock_code)
        if stock_data:
            reply_content = gemini_stock_analysis(stock_code, stock_data)
        else:
            reply_content = "获取行情数据失败，请检查标的代码是否正确，或稍后重试"

    # 构造微信回复XML
    reply_xml = f"""
    <xml>
        <ToUserName><![CDATA[{from_user}]]></ToUserName>
        <FromUserName><![CDATA[{to_user}]]></FromUserName>
        <CreateTime>{int(os.time())}</CreateTime>
        <MsgType><![CDATA[text]]></MsgType>
        <Content><![CDATA[{reply_content}]]></Content>
    </xml>
    """
    return Response(content=reply_xml, media_type="application/xml")

# -------------------------- OpenClaw兼容接口 --------------------------
@app.post("/v1/chat/completions")
async def openclaw_completions(request: Request):
    """兼容OpenAI格式的接口，供OpenClaw转发调用"""
    body = await request.json()
    prompt = body.get("messages", [])[-1].get("content", "")
    stream = body.get("stream", False)

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            stream=stream
        )

        if stream:
            # 流式响应适配OpenClaw
            async def stream_generator():
                for chunk in response:
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk.text}}]})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        else:
            return {
                "id": "gemini-" + os.urandom(8).hex(),
                "object": "chat.completion",
                "created": int(os.time()),
                "model": GEMINI_MODEL,
                "choices": [{"message": {"role": "assistant", "content": response.text}, "finish_reason": "stop"}]
            }
    except Exception as e:
        print(f"OpenClaw接口调用失败：{e}")
        raise HTTPException(status_code=500, detail="服务调用失败")

# -------------------------- 定时任务（每日行情推送） --------------------------
async def daily_market_push():
    """每日收盘后大盘分析推送，可对接微信群机器人"""
    # 此处可扩展：获取大盘数据，调用Gemini生成分析，通过微信webhook推送
    pass

# -------------------------- 服务启动配置 --------------------------
@app.on_event("startup")
async def startup_event():
    scheduler.add_job(daily_market_push, "cron", hour=15, minute=30)  # A股收盘后执行
    scheduler.start()

@app.on_event("shutdown")
async def shutdown_event():
    scheduler.shutdown()

@app.get("/")
async def root():
    return {"status": "running", "message": "股票基金预测Gemini AI助手部署成功"}

if __name__ == "__main__":
    # 适配Zeabur的PORT环境变量
    port = int(os.getenv("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
