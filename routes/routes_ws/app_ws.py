""" 앱 클라이언트 WebSocket 핸들러 (topic 구독) """
import asyncio
from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect
from .utils import maybe_set_base_url_from_ws, log_exc, app_add, app_remove

router = APIRouter()

@router.websocket("/ws/app")
async def ws_app(websocket: WebSocket):
    """
    앱 클라이언트 연결 관리 (topic 기반 구독)
    Query params:
        - topic: 구독할 토픽 (기본값: "public")
    """
    try:
        await websocket.accept()
    except Exception as e:
        log_exc("[APP accept]", e)
        return

    maybe_set_base_url_from_ws(websocket)
    topic = websocket.query_params.get("topic", "public")
    await app_add(topic, websocket)
    print(f"📱 앱 연결됨: topic={topic}")

    try:
        while True:
            try:
                # 클라이언트로부터 메시지 수신 (60초 타임아웃)
                _ = await asyncio.wait_for(websocket.receive_text(), timeout=60)
            except asyncio.TimeoutError:
                continue
            except WebSocketDisconnect:
                break
            except Exception as e:
                log_exc("[APP recv]", e)
                break
    finally:
        try:
            await app_remove(topic, websocket)
            await websocket.close()
        except Exception:
            pass
        print(f"📱 앱 연결 끊김: topic={topic}")