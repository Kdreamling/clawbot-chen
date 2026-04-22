"""
魔改版 dusapi.py —— 指向 Reverie Gateway（SSE 流式 + 智能 channel 选择）
- 启动时 login 拿 JWT，过期自动重登
- /v1/chat/completions 走 stream=True，读 SSE 聚合 text_delta
- Gateway SSE 已天然分离 thinking/text，这边只保留 text
- 智能模型选择：定期调 /admin/channels，按 preferences 挑第一个 enabled channel 的 model
- 保留原 DusConfig/DusAPI 类名，bot.py 无需大改
"""
import json
import re
import time
import uuid
import requests
from dataclasses import dataclass, field
from typing import List, Tuple

version = "3.0.0-chen"

# session_id 必须是合法 UUID（Reverie 表列约束）
_SESSION_NS = uuid.UUID("00000000-0000-0000-0000-000000000001")

# 刷新 enabled 通道列表的间隔
_CHANNELS_REFRESH_INTERVAL = 300  # 5 分钟


def log(message, level="INFO"):
    print(f"[{level}] {message}")


def session_name_to_uuid(name: str) -> str:
    return str(uuid.uuid5(_SESSION_NS, name))


@dataclass
class DusConfig:
    api_key: str                           # 保留字段兼容 bot.py（未使用）
    base_url: str                          # 例：http://localhost:8001
    model1: str = "opus46-ciwei-aws"       # 兜底模型（首次 pick 前用）
    prompt: str = ""
    session_id: str = "wechat-chen-dream"
    auth_password: str = ""
    # [(model_name, channel_name), ...] 按"优先 → 兜底"顺序
    model_preferences: List[Tuple[str, str]] = field(default_factory=list)


class DusAPI:
    def __init__(self, config: DusConfig):
        self.config = config
        self.DS_NOW_MOD = config.model1
        self.base_url = config.base_url.rstrip('/')
        sid = config.session_id
        try:
            uuid.UUID(sid)
            self._session_uuid = sid
        except (ValueError, AttributeError):
            self._session_uuid = session_name_to_uuid(sid or "wechat-chen-dream")
        self._token = None
        self._token_expire = 0
        self._last_channel_refresh = 0
        log(f"session_id: {config.session_id} -> {self._session_uuid}")
        # 启动时尝试按偏好选一个
        self._pick_model_if_needed(force=True)

    def _login(self):
        url = f"{self.base_url}/api/auth/login"
        r = requests.post(url, json={"password": self.config.auth_password}, timeout=15)
        r.raise_for_status()
        data = r.json()
        self._token = data["token"]
        self._token_expire = data.get("expires_at", time.time() + 6 * 86400) - 86400
        log(f"Gateway 登录成功，token 有效至 {time.strftime('%Y-%m-%d %H:%M', time.localtime(self._token_expire))}")

    def _ensure_token(self):
        if not self._token or time.time() >= self._token_expire:
            self._login()

    def _pick_model_if_needed(self, force=False):
        """每 5 分钟或 force=True 时，按 preferences 刷新当前模型选择。"""
        now = time.time()
        if not force and (now - self._last_channel_refresh < _CHANNELS_REFRESH_INTERVAL):
            return
        if not self.config.model_preferences:
            return  # 没配偏好，保持兜底
        try:
            self._ensure_token()
            r = requests.get(
                f"{self.base_url}/api/admin/channels",
                headers={"Authorization": f"Bearer {self._token}"},
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
            enabled = {ch["name"] for ch in data.get("channels", []) if ch.get("enabled")}
            for pref in self.config.model_preferences:
                model, channel = pref[0], pref[1]
                if channel in enabled:
                    if self.DS_NOW_MOD != model:
                        log(f"智能切换模型: {self.DS_NOW_MOD} -> {model} (channel={channel})")
                    self.DS_NOW_MOD = model
                    self._last_channel_refresh = now
                    return
            log("偏好列表里没有任何 channel 处于启用状态，保持当前模型", "WARN")
            self._last_channel_refresh = now
        except Exception as e:
            log(f"刷新 channel 列表失败: {e}", "WARN")

    def _stream_chat(self, model, message):
        """发一条消息，读 SSE 流，返回正文（不含 thinking）。"""
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": message}],
            "stream": True,
        }
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "X-Session-Id": self._session_uuid,
            "Authorization": f"Bearer {self._token}",
            "X-Client-Hint": "wechat",
        }
        text_parts = []
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=180) as r:
            if r.status_code == 401:
                raise RuntimeError("token invalid")
            r.raise_for_status()
            for raw in r.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data: "):
                    continue
                data_str = raw[6:]
                if data_str == "[DONE]":
                    break
                try:
                    ev = json.loads(data_str)
                except Exception:
                    continue
                et = ev.get("type")
                if et == "text_delta":
                    text_parts.append(ev.get("content", ""))
                elif et == "error":
                    raise RuntimeError(f"Gateway 错误: {ev.get('message', 'unknown')}")
                # thinking_* / tool_* / done 忽略
        return "".join(text_parts).strip()

    def chat(self, message, model=None, stream=False, prompt=None, history=None):
        # 每次对话前按需刷新模型选择（内部有 TTL 保护）
        self._pick_model_if_needed()
        model = model or self.DS_NOW_MOD
        retry_delays = [2, 4, 8, 16, 32]
        max_retries = 5
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                self._ensure_token()
                text = self._stream_chat(model, message)
                if not text:
                    text = "（晨没说话）"
                if attempt > 0:
                    log(f"Gateway 第 {attempt} 次重试成功：{text[:80]}...")
                else:
                    log(f"Gateway 返回（model={model}）：{text[:80]}...")
                return text
            except RuntimeError as e:
                if "token invalid" in str(e):
                    self._token = None
                last_error = e
            except Exception as e:
                last_error = e

            if attempt < max_retries:
                delay = retry_delays[attempt]
                log(f"Gateway 第 {attempt+1} 次失败（{type(last_error).__name__}: {last_error}），{delay}s 后重试", "WARN")
                time.sleep(delay)
            else:
                log(f"Gateway 已重试 {max_retries} 次最终失败：{last_error}", "ERROR")

        return "网络抖了一下，晨暂时没收到。过几分钟再试试。"
