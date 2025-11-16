"""ESP32 WebSocket 핸들러 (Heartbeat & 명령)"""
import asyncio
from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from runtime import DEFAULT_ESP32_ID, esp32_conns
from .utils import maybe_set_base_url_from_ws, log_exc

router = APIRouter()

@router.websocket("/ws/esp32")
async def ws_esp32(websocket: WebSocket):
    """
    ESP32 연결 관리 (Heartbeat + 명령 수신)
    Query params:
        - id: ESP32 ID (기본값: DEFAULT_ESP32_ID)
    """
    # 🔍 1) 누가 접속을 시도했는지 무조건 찍기
    print("[ESP32] incoming WS:", websocket.client)

    try:
        await websocket.accept()
        print("[ESP32] accept OK")
    except Exception as e:
        log_exc("[ESP32 accept]", e)
        return

    maybe_set_base_url_from_ws(websocket)
    esp_id = websocket.query_params.get("id", DEFAULT_ESP32_ID)
    esp32_conns[esp_id] = websocket
    print(f"🔌 ESP32 connected: {esp_id} (total={len(esp32_conns)})")

    # 연결 직후 인사 한 번 보내보기 (ESP에서 이게 보이면 완전 성공)
    try:
        await websocket.send_text('{"t":"hello","msg":"esp32 connected"}')
    except Exception as e:
        log_exc("[ESP32 send hello]", e)

    async def _recv_loop():
        """메시지 수신 루프"""
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=60)
                print(f"[ESP32 rx] {msg}")
            except asyncio.TimeoutError:
                # 60초 동안 아무것도 안 와도 계속 유지
                continue
            except WebSocketDisconnect as e:
                # 🔍 왜 끊겼는지 코드/이유 로그
                print(f"[ESP32] WebSocketDisconnect code={e.code} reason={e.reason}")
                break
            except Exception as e:
                log_exc("[ESP32 recv]", e)
                break

    async def _ping_loop():
        """Keep-alive 핑 루프"""
        while True:
            try:
                await asyncio.sleep(15)
                # 그냥 살아있는지만 확인하는 keep-alive 메시지
                await websocket.send_text('{"t":"ping"}')
                print(f"[ESP32] ping -> {esp_id}")
            except Exception as e:
                log_exc("[ESP32 ping]", e)
                break

    recv_task = asyncio.create_task(_recv_loop())
    ping_task = asyncio.create_task(_ping_loop())
    try:
        await asyncio.wait({recv_task, ping_task}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for t in (recv_task, ping_task):
            try:
                t.cancel()
            except:
                pass
        try:
            if esp32_conns.get(esp_id) is websocket:
                esp32_conns.pop(esp_id, None)
            await websocket.close()
        except Exception:
            pass
        print(f"🔌 ESP32 disconnected: {esp_id} (total={len(esp32_conns)})")
