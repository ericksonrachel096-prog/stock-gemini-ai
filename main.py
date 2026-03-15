import os
import hashlib
import xmltodict
import requests
import json
import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from google import genai
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# 加载环境变量
load_dotenv()
app = FastAPI(title="🦞 AI龙虾智能股票助手")
app.mount("/static", StaticFiles(directory="static"), name="static")
scheduler = AsyncIOScheduler(timezone=os.getenv("TZ", "Asia/Shanghai"))

# -------------------------- 核心配置 --------------------------
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL = "gemini-2.0-flash"

# 股票数据源
STOCK_API_KEY = os.getenv("STOCK_API_KEY")
STOCK_API_URL = "https://www.alphavantage.co/query"
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN")
ts.set_token(TUSHARE_TOKEN)
tushare_pro = ts.pro_api()

# 微信配置
WECHAT_TOKEN = os.getenv("WECHAT_TOKEN")

# 简单的内存存储：用于“学习”用户历史对话（生产环境可换Redis）
user_history = {}

# -------------------------- 1. 技术指标计算模块（增强预测能力） --------------------------
def calculate_technical_indicators(df: pd.DataFrame) -> dict:
    """
    计算经典技术指标，为AI预测提供数据支撑：
    - MA5/MA10/MA20（移动平均线）
    - MACD（趋势指标）
    - RSI（超买超卖指标）
    """
    if df.empty or len(df) < 30:
        return {}

    # 按日期升序排列，方便计算
    df = df.sort_values('trade_date').reset_index(drop=True)
    close = df['close']
    
    indicators = {}
    
    # 1. 移动平均线 (MA)
    indicators['MA5'] = close.rolling(5).mean().iloc[-1]
    indicators['MA10'] = close.rolling(10).mean().iloc[-1]
    indicators['MA20'] = close.rolling(20).mean().iloc[-1]
    
    # 2. MACD
    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=9, adjust=False).mean()
    indicators['MACD'] = macd.iloc[-1]
    indicators['MACD_Signal'] = signal.iloc[-1]
    indicators['MACD_Hist'] = (macd - signal).iloc[-1]
    
    # 3. RSI (14日)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    indicators['RSI'] = 100 - (100 / (1 + rs)).iloc[-1]
    
    # 4. 近期高低点
    indicators['近20日最高'] = df['high'].iloc[-20:].max()
    indicators['近20日最低'] = df['low'].iloc[-20:].min()
    
    return indicators

# -------------------------- 2. 双数据源智能获取 --------------------------
def get_stock_data(stock_code: str):
    a_stock_prefix = ("600", "000", "300", "688", "001", "002", "003", "301", "601", "603", "605")
    
    # A股逻辑
    if stock_code.startswith(a_stock_prefix):
        try:
            ts_code = f"{stock_code}.SH" if stock_code.startswith(("600", "688", "601", "603", "605")) else f"{stock_code}.SZ"
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
            
            df = tushare_pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if df.empty:
                return None
            
            # 计算技术指标
            tech_indicators = calculate_technical_indicators(df)
            
            return {
                "标的代码": stock_code,
                "交易所": "上交所" if ".SH" in ts_code else "深交所",
                "最近交易日": df.iloc[0]["trade_date"],
                "最新收盘价": df.iloc[0]["close"],
                "当日涨跌幅": f"{df.iloc[0]['pct_chg']}%",
                "成交量": df.iloc[0]["vol"],
                "技术指标": tech_indicators,
                "近1年数据": df.to_dict(orient="records")
            }
        except Exception as e:
            print(f"Tushare失败：{e}")
            return None
    
    # 美股/其他逻辑
    else:
        try:
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": stock_code,
                "apikey": STOCK_API_KEY,
                "outputsize": "compact"
            }
            response = requests.get(STOCK_API_URL, params=params, timeout=10)
            if response.status_code != 200:
                return None
            data = response.json()
            if "Error Message" in data:
                print(f"Alpha Vantage错误：{data['Error Message']}")
                return None
            if "Time Series (Daily)" not in data:
                return None
            return data
        except Exception as e:
            print(f"Alpha Vantage失败：{e}")
            return None

# -------------------------- 3. 增强版AI分析（含历史学习） --------------------------
def gemini_stock_analysis(stock_code: str, stock_data: dict, user_id: str = "default"):
    """
    增强版分析：
    1. 结合技术指标
    2. 读取该用户历史对话，实现“个性化学习”
    3. 更丰富的Prompt模板
    """
    # 获取历史对话
    history = user_history.get(user_id, [])
    history_str = "\n".join([f"历史查询 {i+1}: {h}" for i, h in enumerate(history[-3:])]) # 只取最近3条
    
    prompt = f"""
    角色：你是一只专业且有趣的AI龙虾机器人，外号龙虾君。你精通技术分析，能结合指标给出客观参考。
    
    用户信息：
    - 用户ID：{user_id}
    - 该用户历史查询：{history_str if history_str else '无历史查询'}

    当前分析标的：{stock_code}
    详细行情数据：{json.dumps(stock_data, ensure_ascii=False, indent=2)}

    输出要求（严格遵守）：
    1. 【技术面解读】：结合提供的MA、MACD、RSI等指标，分析当前趋势（金叉/死叉、超买超卖等）。
    2. 【趋势预测】：给出短期（1-2周）和中期（1-3月）的趋势预判，附概率（如：上涨概率60%），**绝对不做保本承诺**。
    3. 【关键价位】：给出支撑位和压力位参考。
    4. 【龙虾君寄语】：用轻松有趣的语气给出风险提示，偶尔用“钳”“虾虾”谐音梗。
    5. 【固定免责声明】：投资有风险，入市需谨慎。本内容仅为AI技术分析参考，不构成任何投资操作建议。

    语言风格：专业但通俗易懂，不要堆砌术语，像和朋友聊天一样。
    """
    
    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        # 记录本次查询到历史，实现“学习”
        if user_id not in user_history:
            user_history[user_id] = []
        user_history[user_id].append(f"{stock_code} - {datetime.now().strftime('%Y-%m-%d')}")
        return response.text
    except Exception as e:
        print(f"Gemini调用失败：{e}")
        return "哎呀，龙虾君的钳子被数据卡住了！请稍后再试~"

# -------------------------- 4. 微信接口 --------------------------
@app.get("/wechat")
async def wechat_verify(signature: str = Query(...), timestamp: str = Query(...), nonce: str = Query(...), echostr: str = Query(...)):
    tmp_list = sorted([WECHAT_TOKEN, timestamp, nonce])
    tmp_str = "".join(tmp_list).encode("utf-8")
    tmp_sign = hashlib.sha1(tmp_str).hexdigest()
    return int(echostr) if tmp_sign == signature else HTTPException(status_code=403)

@app.post("/wechat")
async def wechat_message(request: Request):
    xml_data = await request.body()
    msg_dict = xmltodict.parse(xml_data)["xml"]
    from_user = msg_dict["FromUserName"] # 用这个作为用户ID，实现个性化
    to_user = msg_dict["ToUserName"]
    content = msg_dict.get("Content", "").strip()

    reply_content = "欢迎光临龙虾君的股票小铺🦞！\n发送格式：【股票代码】即可\n例：600000"

    if content and len(content) >= 4:
        # 简单提取代码（兼容“600000 预测”或直接“600000”）
        stock_code = content.split()[0]
        stock_data = get_stock_data(stock_code)
        if stock_data:
            reply_content = gemini_stock_analysis(stock_code, stock_data, user_id=from_user)
        else:
            reply_content = f"虾虾！没找到代码【{stock_code}】的数据，请检查代码是否正确（如美股需加后缀，如AAPL），或稍后再试~"

    reply_xml = f"""
    <xml>
        <ToUserName><![CDATA[{from_user}]]></ToUserName>
        <FromUserName><![CDATA[{to_user}]]></FromUserName>
        <CreateTime>{int(os.time.get())}</CreateTime>
        <MsgType><![CDATA[text]]></MsgType>
        <Content><![CDATA[{reply_content}]]></Content>
    </xml>
    """
    return Response(content=reply_xml, media_type="application/xml")

# -------------------------- 5. OpenClaw兼容接口 --------------------------
@app.post("/v1/chat/completions")
async def openclaw_completions(request: Request):
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
        print(f"OpenClaw失败：{e}")
        raise HTTPException(status_code=500, detail="服务调用失败")

# -------------------------- 6. 网页端接口（已修复） --------------------------
class PredictRequest(BaseModel):
    stock_code: str

@app.post("/api/predict")
async def api_predict(request: PredictRequest):
    """网页端调用接口"""
    stock_code = request.stock_code.strip()
    stock_data = get_stock_data(stock_code)
    
    if not stock_data:
        # 特别提示Alpha Vantage的Key问题
        if not STOCK_API_KEY:
            return {"status": "error", "message": "请先在环境变量中配置 STOCK_API_KEY (Alpha Vantage)"}
        return {"status": "error", "message": f"获取不到【{stock_code}】的数据，请检查代码或稍后重试"}
    
    try:
        # 网页端用固定ID "web_user"，也可以扩展为登录系统
        result = gemini_stock_analysis(stock_code, stock_data, user_id="web_user")
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "message": f"AI分析出错：{str(e)}"}

# -------------------------- 7. 网页与健康检查 --------------------------
@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.get("/health")
async def health_check():
    return {"status": "running", "message": "🦞 AI龙虾智能股票助手运行中", "tech_indicators_enabled": True}

# -------------------------- 8. 定时任务 --------------------------
async def daily_market_push():
    pass # 预留扩展

@app.on_event("startup")
async def startup_event():
    scheduler.add_job(daily_market_push, "cron", hour=15, minute=30)
    scheduler.start()

@app.on_event("shutdown")
async def shutdown_event():
    scheduler.shutdown()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
