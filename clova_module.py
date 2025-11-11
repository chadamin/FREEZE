# clova_module.py — CLOVA Speech gRPC 실시간 모듈 (DROP-IN)
# ------------------------------------------------
# - gRPC 기반 실시간 STT
# - 입력: 오디오(16kHz/mono/16bit PCM RAW 권장, WAV도 자동 변환)
# - 출력: on_result(text:str, meta:dict) 콜백으로 부분/최종 결과 전달
# ------------------------------------------------

import os, io, wave, json, queue, threading, tempfile, subprocess, time
from urllib.parse import urlparse
import grpc

# ====== proto stubs (nest.proto를 컴파일해 생성된 파일들) ======
import nest_pb2
import nest_pb2_grpc

# ================== ENV ==================
CLOVA_SPEECH_HOST = os.getenv("CLOVA_SPEECH_HOST", "")     # 예: "clovaspeech-gw.ncloud.com:443"
CLOVA_ACCESS_KEY  = os.getenv("CLOVA_ACCESS_KEY", "")      # Authorization: Bearer <THIS>
LANG_CODE         = os.getenv("CLOVA_SPEECH_LANG", "ko-KR")
ENABLE_ITN        = os.getenv("CLOVA_SPEECH_ITN", "true").lower() == "true"
ENABLE_EOS        = os.getenv("CLOVA_SPEECH_EOS", "true").lower() == "true"
CHUNK_MS          = int(os.getenv("CLOVA_SPEECH_CHUNK_MS", "320"))
CLOVA_LOG         = os.getenv("CLOVA_LOG", "1") == "1"

RAW_SR, RAW_CH, RAW_SW = 16000, 1, 2  # 16k/mono/16bit

def _log(*args):
    if CLOVA_LOG:
        print(time.strftime("[%H:%M:%S]"), *args, flush=True)

def _normalize_host(h: str) -> str:
    """스킴이 있든 없든 'host:port' 로 정규화."""
    h = (h or "").strip()
    if not h:
        return "clovaspeech-gw.ncloud.com:443"
    if "://" in h:
        u = urlparse(h)
        host = u.hostname or "clovaspeech-gw.ncloud.com"
        port = u.port or 443
        return f"{host}:{port}"
    return h

# ================== 유틸 ==================
def _ffmpeg_to_raw16le(data: bytes) -> bytes:
    """임의 오디오 -> 16k/mono/s16le RAW"""
    with tempfile.NamedTemporaryFile(suffix=".in", delete=True) as f_in, \
         tempfile.NamedTemporaryFile(suffix=".raw", delete=True) as f_out:
        f_in.write(data); f_in.flush()
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", f_in.name,
            "-f", "s16le", "-acodec", "pcm_s16le",
            "-ar", str(RAW_SR), "-ac", str(RAW_CH),
            f_out.name
        ]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p.returncode != 0:
            raise RuntimeError("ffmpeg convert failed")
        return open(f_out.name, "rb").read()

def _wav_to_raw16le(wav_bytes: bytes) -> bytes:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        assert wf.getframerate() == RAW_SR,  "WAV samplerate must be 16k"
        assert wf.getnchannels()  == RAW_CH, "WAV channels must be mono"
        assert wf.getsampwidth()  == RAW_SW, "WAV must be 16-bit"
        frames = wf.readframes(wf.getnframes())
    return frames

# ================== 스트리머 클래스 ==================
class ClovaSpeechStreamer:
    def __init__(self, on_result=None):
        """
        on_result: 콜백 (text:str, meta:dict)
        """
        self.on_result = on_result or (lambda text, meta: None)
        self._q = queue.Queue(maxsize=256)
        self._stop = threading.Event()
        self._thr = None
        self._channel = None
        self._stub = None
        self._stream_call = None

    # ---- 내부 헬퍼: Stub / 메서드 자동 탐색 ----
    def _pick_stub(self):
        candidates = ["NESTStub", "NestServiceStub", "ClovaSpeechStub", "SpeechStub"]
        for name in candidates:
            stub = getattr(nest_pb2_grpc, name, None)
            if stub:
                return stub
        # 최후 수단: 첫 번째 *_Stub 클래스를 잡는다
        for k, v in vars(nest_pb2_grpc).items():
            if isinstance(v, type) and k.endswith("Stub"):
                return v
        raise RuntimeError("gRPC Stub class not found in nest_pb2_grpc.py. Check service name in nest.proto")

    def _pick_stream_method(self):
        candidates = ["StreamingRecognize", "streamingRecognize", "RecognizeStream", "Stream"]
        for name in candidates:
            fn = getattr(self._stub, name, None)
            if fn:
                return fn
        avail = [k for k in dir(self._stub) if not k.startswith("_")]
        raise RuntimeError(f"Streaming RPC not found in stub. Available: {avail}")

    def start(self):
        assert CLOVA_ACCESS_KEY,  "0f3b7502b985401cb5fc23d87d4a47fe"

        host = _normalize_host(CLOVA_SPEECH_HOST)
        authority = host.split(":")[0]
        _log("[CLOVA] target=", host, "authority=", authority, "access_key=", "set" if CLOVA_ACCESS_KEY else "MISSING")

        # TLS + HTTP/2 (ALPN=h2). authority/SNI 강제
        ssl_creds = grpc.ssl_channel_credentials()
        options = [
            ("grpc.ssl_target_name_override", authority),
            ("grpc.default_authority",        authority),
            ("grpc.max_send_message_length",  50 * 1024 * 1024),
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),
        ]
        self._channel = grpc.secure_channel(host, ssl_creds, options=options)

        # ALPN/TLS 준비 확인 (여기서 실패하면 즉시 로그로 원인 확인 가능)
        try:
            grpc.channel_ready_future(self._channel).result(timeout=5)
            _log("[CLOVA] gRPC TLS/ALPN handshake ready ->", host)
        except Exception as e:
            _log("[CLOVA] handshake FAIL ->", host, e)
            raise

        Stub = self._pick_stub()
        self._stub = Stub(self._channel)
        self._stream_call = self._pick_stream_method()

        self._stop.clear()
        self._thr = threading.Thread(target=self._run, daemon=True)
        self._thr.start()

    def close(self):
        self._stop.set()
        if self._thr:
            self._thr.join(timeout=1.0)
        if self._channel:
            self._channel.close()
        self._thr = None
        self._channel = None

    def push_audio(self, chunk: bytes):
        """웹소켓에서 받은 오디오 바이트 그대로 밀어넣기"""
        if chunk:
            self._q.put(chunk)

    # 내부 루프
    def _run(self):
        config = {
            "language": LANG_CODE,
            "sampleRate": RAW_SR,
            "enableItn": ENABLE_ITN,
            "enableEndOfSpeechDetection": ENABLE_EOS,
            "punctuation": True,
            "timestamps": True,
        }

        def gen():
            # 세션 초기 설정
            yield nest_pb2.StreamingRequest(config=json.dumps(config).encode("utf-8"))
            buf = bytearray()
            bytes_per_chunk = RAW_SR * RAW_CH * RAW_SW * CHUNK_MS // 1000
            while not self._stop.is_set():
                try:
                    data = self._q.get(timeout=0.1)
                    raw = self._to_raw(data)
                    buf.extend(raw)
                    while len(buf) >= bytes_per_chunk:
                        frame = bytes(buf[:bytes_per_chunk])
                        del buf[:bytes_per_chunk]
                        yield nest_pb2.StreamingRequest(audio=frame)
                except queue.Empty:
                    continue
            # 종료 신호
            yield nest_pb2.StreamingRequest(eos=True)

        metadata = (("authorization", f"Bearer {CLOVA_ACCESS_KEY}"),)
        try:
            responses = self._stream_call(gen(), timeout=3600, metadata=metadata)
            for resp in responses:
                if getattr(resp, "result", None):
                    try:
                        rj = json.loads(resp.result)
                        text = rj.get("text") or rj.get("transcript") or ""
                        if text:
                            _log("[CLOVA]", ("FINAL" if rj.get("is_final") else "PART"), text)
                            self.on_result(text, rj)
                    except Exception:
                        raw = getattr(resp, "result", "").strip()
                        if raw:
                            _log("[CLOVA]", "RAW", raw)
                            self.on_result(raw, {"raw": raw})
        except grpc.RpcError as e:
            _log("[CLOVA] gRPC error:", e)

    def _to_raw(self, data: bytes) -> bytes:
        if len(data) >= 4 and data[:4] == b"RIFF":  # WAV
            return _wav_to_raw16le(data)
        return _ffmpeg_to_raw16le(data)

# ================== 간단한 헬퍼 함수 ==================
# 맨 아래 헬퍼 영역 바꿔 넣기

_streamer: ClovaSpeechStreamer = None


def start_clova(on_result=None):
    """스트리머 생성 및 시작 (전역 핸들에 보관)"""
    global _streamer
    if _streamer:
        stop_clova()
    s = ClovaSpeechStreamer(on_result=on_result)
    s.start()
    _streamer = s
    return s

def stop_clova():
    global _streamer
    if _streamer:
        _streamer.close()
        _streamer = None

def push_audio(data: bytes):
    """오디오 바이트를 계속 밀어넣기"""
    if _streamer:
        _streamer.push_audio(data)
