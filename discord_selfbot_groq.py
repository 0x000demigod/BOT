import asyncio
import base64
import io
import json
import logging
import os
import random
import time
from collections import defaultdict, deque

import aiohttp
import discord
from bs4 import BeautifulSoup
from ddgs import DDGS
from discord.ext import tasks
from openai import AsyncOpenAI
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("selfbot")

# --- CONFIGURATION ---
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
COOLDOWN_SECONDS = 10
HISTORY_LIMIT = 15

# DM Follow-up Settings
FOLLOWUP_MIN_MINUTES = 30
FOLLOWUP_MAX_MINUTES = 60
CHECK_INTERVAL_MINUTES = 2

# Server Autonomous Chat Settings
AUTONOMOUS_CHANNEL_ID = int(os.getenv("AUTONOMOUS_CHANNEL_ID", "0"))
AUTONOMOUS_INTERVAL_MINUTES = int(os.getenv("AUTONOMOUS_INTERVAL_MINUTES", "120"))

# Voice Note Settings
VOICE_NOTE_CHANCE = 0.4 # 40% chance to reply with a voice note in DMs
GEMINI_TTS_MODEL = "gemini-3.8-flash-tts"
GEMINI_VOICE = "Kore"

SYSTEM_PROMPT = """You are acting on behalf of the Discord account owner. 
You are having a conversation with a friend. Keep replies short, casual, and match the tone of the conversation. Never mention that you are an AI. Sound exactly like a human texting.
You have access to tools: `search_web` and `fetch_url`. Use them when you need real-time information.
"""

# Clients
gemini_client = AsyncOpenAI(
    api_key=os.environ.get("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
tts_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# State management
history = defaultdict(lambda: deque(maxlen=HISTORY_LIMIT))
last_reply = defaultdict(float)
_channel_locks = defaultdict(asyncio.Lock)
dm_sessions = {}

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web using DuckDuckGo. Use this for real-time info.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch the text content of a webpage. Use this when you have a URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL (must start with http)"}
                },
                "required": ["url"]
            }
        }
    }
]

def _ddg_search(query, max_results=3):
    try:
        return DDGS().text(query, max_results=max_results)
    except Exception as e:
        log.error(f"DDG search error: {e}")
        return []

async def search_web(query):
    results = await asyncio.to_thread(_ddg_search, query)
    if not results:
        return "No results found."
    return "\n".join([f"Title: {r.get('title')}\nURL: {r.get('href')}\nSnippet: {r.get('body')}\n" for r in results])

async def fetch_url_content(url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                html = await response.text()
        soup = BeautifulSoup(html, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator=' ', strip=True)
        return text[:4000]
    except Exception as e:
        return f"Failed to fetch URL: {e}"

# --- VOICE NOTE GENERATION & FFMPEG ---
async def convert_to_voice_note(audio_data: bytes) -> io.BytesIO:
    """Converts raw audio bytes to OGG Opus using FFmpeg in memory."""
    command = [
        'ffmpeg', '-y',
        '-i', 'pipe:0',
        '-c:a', 'libopus',
        '-b:a', '64k',
        '-f', 'ogg',
        'pipe:1'
    ]
    
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await process.communicate(input=audio_data)
    
    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {stderr.decode()}")
        
    return io.BytesIO(stdout)

async def generate_and_send_voice_note(text: str, channel: discord.abc.Messageable):
    try:
        log.info("Generating TTS audio...")
        # 1. Call Gemini TTS API
        response = await tts_client.aio.models.generate_content(
            model=GEMINI_TTS_MODEL,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=GEMINI_VOICE)
                    )
                )
            )
        )
        
        # 2. Extract audio data
        audio_data = response.candidates[0].content.parts[0].inline_data.data
        raw_audio_bytes = base64.b64decode(audio_data)

        # 3. Convert to OGG using FFmpeg
        log.info("Converting to OGG via FFmpeg...")
        ogg_buffer = await convert_to_voice_note(raw_audio_bytes)

        # 4. Send to Discord as native voice note
        audio_file = discord.File(ogg_buffer, filename="voice-message.ogg")
        await channel.send(file=audio_file, flags=discord.MessageFlags(8192))
        log.info("Voice note sent successfully!")
        
    except Exception as e:
        log.error(f"Failed to send voice note: {e}")
        # Fallback to text
        await channel.send(text[:2000])


class SelfBot(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def setup_hook(self):
        self.dm_checker.start()
        if AUTONOMOUS_CHANNEL_ID > 0:
            self.server_chat_loop.start()

    async def on_ready(self):
        log.info("Logged in as %s (self-bot) — ToS risk accepted by operator", self.user)

    @tasks.loop(minutes=CHECK_INTERVAL_MINUTES)
    async def dm_checker(self):
        now = time.time()
        for channel_id, session in list(dm_sessions.items()):
            elapsed_mins = (now - session["last_active"]) / 60
            
            if FOLLOWUP_MIN_MINUTES <= elapsed_mins <= FOLLOWUP_MAX_MINUTES and not session["followed_up"]:
                channel = self.get_channel(channel_id)
                if not channel: continue
                
                session["followed_up"] = True
                convo = list(history[channel_id])
                if not convo: continue
                
                log.info(f"Autonomously continuing DM conversation in {channel_id}...")
                follow_up_instruct = "Based on our chat history, what should I say next to keep the conversation going? Just reply with the message."
                
                try:
                    response = await gemini_client.chat.completions.create(
                        model=MODEL,
                        max_tokens=150,
                        messages=[{"role": "system", "content": SYSTEM_PROMPT}, *convo, {"role": "user", "content": follow_up_instruct}]
                    )
                    msg = response.choices[0].message.content
                    
                    if msg and msg.strip():
                        async with channel.typing():
                            await asyncio.sleep(3)
                            # Follow-ups are always text for simplicity
                            await channel.send(msg.strip())
                        history[channel_id].append({"role": "assistant", "content": msg.strip()})
                        log.info("Autonomous DM message sent.")
                except Exception as e:
                    log.error(f"Autonomous DM follow-up failed: {e}")

    @tasks.loop(minutes=AUTONOMOUS_INTERVAL_MINUTES)
    async def server_chat_loop(self):
        if AUTONOMOUS_CHANNEL_ID <= 0: return
        channel = self.get_channel(AUTONOMOUS_CHANNEL_ID)
        if not channel: return
        
        prompt = "Generate a short, casual conversation starter or an interesting thought to send to a friend on Discord. Do not use quotes."
        try:
            response = await gemini_client.chat.completions.create(
                model=MODEL,
                max_tokens=100,
                messages=[{"role": "system", "content": "You are a casual Discord user."}, {"role": "user", "content": prompt}]
            )
            msg = response.choices[0].message.content
            if msg and msg.strip():
                async with channel.typing():
                    await asyncio.sleep(2)
                    await channel.send(msg.strip())
        except Exception as e:
            log.error(f"Server autonomous task failed: {e}")

    async def on_message(self, message: discord.Message):
        if message.author.bot or not message.content or not message.content.strip():
            return

        in_server = message.guild is not None
        channel_id = message.channel.id

        if not in_server:
            if channel_id not in dm_sessions:
                dm_sessions[channel_id] = {"followed_up": False}
            dm_sessions[channel_id]["last_active"] = time.time()
            dm_sessions[channel_id]["followed_up"] = False 

        if message.author.id == self.user.id:
            if not in_server:
                history[channel_id].append({"role": "assistant", "content": message.content})
            return 

        if in_server and self.user not in message.mentions:
            return

        now = time.time()
        if now - last_reply[channel_id] < COOLDOWN_SECONDS:
            return
        last_reply[channel_id] = now

        raw = message.content
        raw = raw.replace(f"<@{self.user.id}>", "").replace(f"<@!{self.user.id}>", "")
        raw = raw.replace(f"@{self.user.display_name}", "")
        text = raw.strip()
        if not text: return

        hist_key = (channel_id, message.author.id) if in_server else channel_id
        convo = list(history[hist_key])
        convo.append({"role": "user", "content": f"{message.author.display_name}: {text}"})

        async with _channel_locks[channel_id]:
            async with message.channel.typing():
                reply = ""
                try:
                    try:
                        response = await gemini_client.chat.completions.create(
                            model=MODEL,
                            max_tokens=400,
                            messages=[{"role": "system", "content": SYSTEM_PROMPT}, *convo],
                            tools=tools,
                            tool_choice="auto"
                        )
                        response_message = response.choices[0].message
                        
                        if response_message.tool_calls:
                            convo.append(response_message)
                            for tool_call in response_message.tool_calls:
                                try:
                                    args = json.loads(tool_call.function.arguments)
                                    query = args.get("query", text)
                                except:
                                    query = text
                                
                                result = await search_web(query)
                                convo.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "name": "search_web",
                                    "content": result
                                })
                            
                            final_response = await gemini_client.chat.completions.create(
                                model=MODEL,
                                max_tokens=400,
                                messages=[{"role": "system", "content": SYSTEM_PROMPT}, *convo]
                            )
                            reply = final_response.choices[0].message.content or ""
                        else:
                            reply = response_message.content or ""
                            
                    except Exception as tool_error:
                        log.warning(f"Tool calling failed. Falling back to manual search.")
                        search_results = await search_web(text)
                        fallback_prompt = f"{SYSTEM_PROMPT}\n\nWeb context:\n{search_results}"
                        fallback_response = await gemini_client.chat.completions.create(
                            model=MODEL,
                            max_tokens=400,
                            messages=[{"role": "system", "content": fallback_prompt}, *convo]
                        )
                        reply = fallback_response.choices[0].message.content or ""

                except Exception as e:
                    log.exception(f"All API attempts failed: {e}")
                    return

            if not reply.strip(): return

            convo.append({"role": "assistant", "content": reply})
            history[hist_key].clear()
            history[hist_key].extend(convo[-HISTORY_LIMIT:])

            # --- DECIDE: TEXT OR VOICE NOTE? ---
            # In DMs, roll the dice based on VOICE_NOTE_CHANCE. In servers, always reply with text.
            if not in_server and random.random() < VOICE_NOTE_CHANCE:
                await generate_and_send_voice_note(reply, message.channel)
            else:
                await message.reply(reply[:2000], mention_author=False)


if __name__ == "__main__":
    SelfBot().run(os.environ["DISCORD_USER_TOKEN"])