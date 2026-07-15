"""Authenticated stdlib HTTP surface for the Capture Studio backend."""
# ruff: noqa: E501, RUF001

from __future__ import annotations

import base64
import hmac
import json
import ssl
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import BinaryIO, cast
from urllib.parse import parse_qs, urlsplit

from pydantic import BaseModel, ConfigDict, Field

from nana_tracking.data.capture import CaptureChunk
from nana_tracking.data.studio import (
    CaptureQualitySample,
    CaptureStudio,
    ControlAction,
    PreviewMetadata,
    StudioSessionDefinition,
)

MAX_JSON_BYTES = 256 * 1024
MAX_PREVIEW_BYTES = 4 * 1024 * 1024
MAX_CHUNK_BYTES = 512 * 1024 * 1024


class RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ControlRequest(RequestModel):
    action: ControlAction
    take_id: str | None = None
    action_script_id: str | None = None
    retake_of: str | None = None


class CommandAckRequest(RequestModel):
    revision: int = Field(gt=0)
    command_id: str
    device_id: str
    applied_at_ns: int = Field(ge=0)


class CaptureStudioHTTPServer(ThreadingHTTPServer):
    studio: CaptureStudio
    token: str | None

    def __init__(
        self,
        server_address: tuple[str, int],
        studio: CaptureStudio,
        token: str | None,
    ) -> None:
        super().__init__(server_address, CaptureStudioRequestHandler)
        self.studio = studio
        self.token = token


class CaptureStudioRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        if parsed.path == "/":
            self._send_bytes(HTTPStatus.OK, STUDIO_HTML.encode(), "text/html; charset=utf-8")
            return
        if not self._authorized():
            return
        try:
            if parsed.path == "/api/state":
                self._send_json(HTTPStatus.OK, self._studio.state().model_dump(mode="json"))
            elif parsed.path == "/api/commands":
                query = parse_qs(parsed.query)
                after = int(query.get("after", ["0"])[0])
                commands = self._studio.commands_after(after)
                self._send_json(
                    HTTPStatus.OK,
                    [command.model_dump(mode="json") for command in commands],
                )
            elif parsed.path == "/api/receiver-index":
                self._send_json(
                    HTTPStatus.OK,
                    [item.model_dump(mode="json") for item in self._studio.receiver_index()],
                )
            elif parsed.path == "/api/preview":
                preview = self._studio.preview_bytes()
                metadata = self._studio.preview_metadata()
                if preview is None or metadata is None:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "preview unavailable"})
                    return
                self._send_bytes(
                    HTTPStatus.OK,
                    preview,
                    "image/jpeg",
                    extra_headers={
                        "Cache-Control": "no-store",
                        "X-Nana-Preview": _urlsafe_json(metadata.model_dump(mode="json")),
                    },
                )
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "route not found"})
        except (TypeError, ValueError) as error:
            self._send_json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(error)})

    def do_POST(self) -> None:
        parsed = urlsplit(self.path)
        if not self._authorized():
            return
        try:
            if parsed.path == "/api/session":
                definition = StudioSessionDefinition.model_validate(self._read_json())
                server = self._capture_server
                server.studio = CaptureStudio.create(server.studio.root, definition)
                self._send_json(
                    HTTPStatus.CREATED,
                    server.studio.state().model_dump(mode="json"),
                )
            elif parsed.path == "/api/control":
                request = ControlRequest.model_validate(self._read_json())
                command = self._studio.issue_control(
                    request.action,
                    take_id=request.take_id,
                    action_script_id=request.action_script_id,
                    retake_of=request.retake_of,
                )
                self._send_json(HTTPStatus.CREATED, command.model_dump(mode="json"))
            elif parsed.path == "/api/command-ack":
                request = CommandAckRequest.model_validate(self._read_json())
                acknowledgement = self._studio.acknowledge_command(
                    revision=request.revision,
                    command_id=request.command_id,
                    device_id=request.device_id,
                    applied_at_ns=request.applied_at_ns,
                )
                self._send_json(HTTPStatus.OK, acknowledgement.model_dump(mode="json"))
            elif parsed.path == "/api/quality":
                result = self._studio.publish_quality(
                    CaptureQualitySample.model_validate(self._read_json())
                )
                self._send_json(HTTPStatus.OK, result.model_dump(mode="json"))
            elif parsed.path == "/api/finalize":
                manifest = self._studio.finalize_receiver_session()
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "session_id": manifest.session_id,
                        "chunk_count": len(manifest.chunks),
                        "manifest_sha256": manifest.manifest_sha256,
                        "manifest": str(self._studio.root / "receiver" / "session.json"),
                    },
                )
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "route not found"})
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            self._send_json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(error)})

    def do_PUT(self) -> None:
        parsed = urlsplit(self.path)
        if not self._authorized():
            return
        try:
            if parsed.path == "/api/preview":
                body = self._read_body(MAX_PREVIEW_BYTES)
                metadata_header = self.headers.get("X-Nana-Preview")
                if metadata_header is None:
                    raise ValueError("preview metadata header is required")
                metadata = PreviewMetadata.model_validate(_decode_urlsafe_json(metadata_header))
                self._studio.publish_preview(metadata, body)
                self._send_json(HTTPStatus.OK, metadata.model_dump(mode="json"))
            elif parsed.path.startswith("/api/chunks/"):
                chunk_id = parsed.path.removeprefix("/api/chunks/")
                descriptor_header = self.headers.get("X-Nana-Chunk")
                if descriptor_header is None:
                    raise ValueError("chunk descriptor header is required")
                chunk = CaptureChunk.model_validate(_decode_urlsafe_json(descriptor_header))
                if chunk.chunk_id != chunk_id:
                    raise ValueError("chunk route and descriptor IDs differ")
                content_length = self._content_length(MAX_CHUNK_BYTES)
                if content_length != chunk.byte_length:
                    raise ValueError("chunk content length does not match its descriptor")
                acknowledgement = self._studio.receive_chunk(chunk, cast(BinaryIO, self.rfile))
                self._send_json(HTTPStatus.OK, acknowledgement.model_dump(mode="json"))
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "route not found"})
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            self.close_connection = True
            self._send_json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(error)})

    @property
    def _capture_server(self) -> CaptureStudioHTTPServer:
        return cast(CaptureStudioHTTPServer, self.server)

    @property
    def _studio(self) -> CaptureStudio:
        return self._capture_server.studio

    def _authorized(self) -> bool:
        token = self._capture_server.token
        if token is None:
            return True
        authorization = self.headers.get("Authorization", "")
        expected = f"Bearer {token}"
        if hmac.compare_digest(authorization, expected):
            return True
        self._send_json(
            HTTPStatus.UNAUTHORIZED,
            {"error": "authorization required"},
            extra_headers={"WWW-Authenticate": "Bearer"},
        )
        return False

    def _read_json(self) -> object:
        return json.loads(self._read_body(MAX_JSON_BYTES))

    def _read_body(self, maximum: int) -> bytes:
        length = self._content_length(maximum)
        body = self.rfile.read(length)
        if len(body) != length:
            raise ValueError("request body ended before Content-Length")
        return body

    def _content_length(self, maximum: int) -> int:
        raw = self.headers.get("Content-Length")
        if raw is None:
            raise ValueError("Content-Length is required")
        length = int(raw)
        if length < 0 or length > maximum:
            raise ValueError("request body exceeds the configured limit")
        return length

    def _send_json(
        self,
        status: HTTPStatus,
        payload: object,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self._send_bytes(
            status,
            (json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n").encode(),
            "application/json",
            extra_headers=extra_headers,
        )

    def _send_bytes(
        self,
        status: HTTPStatus,
        payload: bytes,
        content_type: str,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'",
        )
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        return


def make_capture_studio_server(
    root: Path,
    *,
    host: str,
    port: int,
    token: str | None,
    tls_cert: Path | None = None,
    tls_key: Path | None = None,
) -> CaptureStudioHTTPServer:
    remote = host not in {"127.0.0.1", "::1", "localhost"}
    if remote and (not token or tls_cert is None or tls_key is None):
        raise ValueError("non-loopback Studio binding requires a bearer token and TLS certificate")
    if (tls_cert is None) != (tls_key is None):
        raise ValueError("TLS certificate and key must be provided together")
    server = CaptureStudioHTTPServer((host, port), CaptureStudio(root), token)
    if tls_cert is not None and tls_key is not None:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(tls_cert, tls_key)
        server.socket = context.wrap_socket(server.socket, server_side=True)
    return server


def _urlsafe_json(payload: object) -> str:
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return base64.urlsafe_b64encode(encoded).rstrip(b"=").decode()


def _decode_urlsafe_json(value: str) -> object:
    encoded = value.encode()
    padding = b"=" * (-len(encoded) % 4)
    return json.loads(base64.b64decode(encoded + padding, altchars=b"-_", validate=True))


STUDIO_HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Nana Capture Studio</title><style>
:root{color-scheme:dark;--bg:#11141a;--card:#1b2029;--line:#303745;--accent:#7dd3fc;--bad:#fb7185}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#edf2f7;font:15px system-ui,sans-serif}
main{max-width:1180px;margin:auto;padding:28px}.grid{display:grid;grid-template-columns:1.4fr 1fr;gap:18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;margin-bottom:18px}
h1{font-size:25px;margin:0 0 20px}h2{font-size:17px;margin:0 0 14px}.row{display:flex;gap:10px;flex-wrap:wrap}
input,button{border:1px solid var(--line);border-radius:8px;padding:10px;background:#12161d;color:inherit}
input{min-width:180px;flex:1}button{cursor:pointer}button.primary{background:#075985;border-color:#0ea5e9}
button.danger{border-color:var(--bad)}img{width:100%;aspect-ratio:16/10;object-fit:contain;background:#090b0f;border-radius:9px}
.preview{position:relative}.preview canvas{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
#curves{position:static;width:100%;height:180px;background:#12161d;border-radius:9px;margin-top:12px}
.status{display:grid;grid-template-columns:repeat(3,1fr);gap:9px}.metric{background:#12161d;border-radius:8px;padding:10px}
.metric span{display:block;color:#94a3b8;font-size:12px}.flags{color:var(--bad)}#notice{min-height:22px;color:var(--accent)}
@media(max-width:780px){.grid{grid-template-columns:1fr}.status{grid-template-columns:1fr 1fr}}
</style></head><body><main><h1>Nana Capture Studio</h1><div id="notice"></div>
<section id="setup" class="card" hidden><h2>创建采集会话</h2><div class="row">
<input id="session" placeholder="会话 ID"><input id="subject" placeholder="受试者 ID"><input id="device" placeholder="设备 ID">
<input id="model" placeholder="设备型号"><input id="os" placeholder="系统版本"><input id="mapping" placeholder="映射版本">
<input id="consent" placeholder="同意记录 ID"><input id="licenses" placeholder="许可记录 ID，逗号分隔">
<button class="primary" onclick="createSession()">创建</button></div></section>
<div id="workspace" class="grid" hidden><div><section class="card"><h2>实时预览</h2><div class="preview"><img id="preview" alt="实时预览"><canvas id="mesh"></canvas></div><canvas id="curves"></canvas></section>
<section class="card"><h2>采集控制</h2><div class="row"><input id="take" placeholder="Take ID"><input id="script" placeholder="动作脚本 ID">
<input id="retake" placeholder="替换的 Take ID"><button class="primary" onclick="control('start')">开始 / 继续</button>
<button onclick="control('pause')">暂停</button><button onclick="control('stop')">结束 Take</button>
<button onclick="control('retake')">重录</button><button class="danger" onclick="control('end')">结束会话</button></div></section></div>
<aside><section class="card"><h2>会话状态</h2><div class="status"><div class="metric"><span>状态</span><b id="state">-</b></div>
<div class="metric"><span>当前 Take</span><b id="current">-</b></div><div class="metric"><span>已接收 Chunk</span><b id="chunks">0</b></div>
<div class="metric"><span>控制版本</span><b id="revision">0</b></div><div class="metric"><span>设备确认</span><b id="acked">0</b></div>
<div class="metric"><span>质量</span><b id="quality">等待</b></div></div><p id="flags" class="flags"></p></section>
<section class="card"><h2>Take 列表</h2><div id="takes"></div></section></aside></div></main><script>
let token=sessionStorage.getItem('nana-token')||'',history={};if(!token){token=prompt('访问口令（本机无口令可留空）','')||'';sessionStorage.setItem('nana-token',token)}
const headers=()=>token?{'Authorization':'Bearer '+token}:{};
async function api(path,options={}){options.headers={...headers(),...(options.headers||{})};let r=await fetch(path,options);let v=await r.json();if(!r.ok)throw Error(v.error||'请求失败');return v}
function notice(v,bad=false){let e=document.getElementById('notice');e.textContent=v;e.style.color=bad?'var(--bad)':'var(--accent)'}
async function refresh(){try{let s=await api('/api/state');setup.hidden=true;workspace.hidden=false;state.textContent=s.status;current.textContent=s.current_take_id||'-';chunks.textContent=s.received_chunk_count;revision.textContent=s.last_command_revision;acked.textContent=s.acknowledged_revision;
let q=s.latest_quality;quality.textContent=q?(q.acceptable?'通过':'需检查'):'等待';flags.textContent=q?q.flags.join(' · '):'';takes.innerHTML=s.takes.map(t=>`<p>${t.take_id} · ${t.action_script_id} · ${t.status}</p>`).join('');drawQuality(q);await previewImage()}catch(e){if(String(e).includes('session has not been created')){setup.hidden=false;workspace.hidden=true}else notice(e.message,true)}}
async function previewImage(){let r=await fetch('/api/preview',{headers:headers(),cache:'no-store'});if(r.ok){let b=await r.blob();let old=preview.src;preview.src=URL.createObjectURL(b);if(old.startsWith('blob:'))URL.revokeObjectURL(old)}}
function drawQuality(q){let mc=mesh,ctx=mc.getContext('2d'),box=preview.getBoundingClientRect();mc.width=Math.max(1,box.width*devicePixelRatio);mc.height=Math.max(1,box.height*devicePixelRatio);ctx.clearRect(0,0,mc.width,mc.height);if(q){ctx.fillStyle='#7dd3fc';for(let p of q.sample.face_mesh){ctx.beginPath();ctx.arc(p.x*mc.width,p.y*mc.height,1.5*devicePixelRatio,0,7);ctx.fill()}for(let [k,v] of Object.entries(q.sample.parameters)){(history[k]??=[]).push(v);if(history[k].length>120)history[k].shift()}}drawCurves()}
function drawCurves(){let c=curves,ctx=c.getContext('2d'),box=c.getBoundingClientRect();c.width=Math.max(1,box.width*devicePixelRatio);c.height=Math.max(1,180*devicePixelRatio);ctx.clearRect(0,0,c.width,c.height);let colors=['#7dd3fc','#fbbf24','#a78bfa','#34d399'],keys=Object.keys(history).sort().slice(0,4);ctx.font=`${12*devicePixelRatio}px system-ui`;keys.forEach((k,i)=>{let a=history[k];ctx.strokeStyle=colors[i];ctx.beginPath();a.forEach((v,n)=>{let x=n/119*c.width,y=(1-Math.max(0,Math.min(1,v)))*c.height;(n?ctx.lineTo(x,y):ctx.moveTo(x,y))});ctx.stroke();ctx.fillStyle=colors[i];ctx.fillText(k,8*devicePixelRatio,(16+i*16)*devicePixelRatio)})}
async function createSession(){try{let body={session_id:session.value,subject_id:subject.value,device_id:device.value,device_model:model.value,os_version:os.value,ntp_mapping_revision:mapping.value,consent_record_id:consent.value,license_record_ids:licenses.value.split(',').map(x=>x.trim()).filter(Boolean),created_at_ns:Date.now()*1000000};await api('/api/session',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});notice('会话已创建');refresh()}catch(e){notice(e.message,true)}}
async function control(action){try{let body={action:action,take_id:['start','pause','stop','retake'].includes(action)?take.value:null,action_script_id:['start','retake'].includes(action)?script.value:null,retake_of:action==='retake'?retake.value:null};await api('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});notice('控制指令已下发');refresh()}catch(e){notice(e.message,true)}}
refresh();setInterval(refresh,1000);
</script></body></html>"""
