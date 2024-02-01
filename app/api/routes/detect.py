from fastapi import APIRouter, UploadFile, File, Header
from time import time
from starlette.requests import Request
from starlette.routing import Route
from app.api.schemas.response import ASRResult
from app.engine import STT
from pydub import AudioSegment
from pydub.effects import normalize
import logging
import traceback
from io import BytesIO

router = APIRouter()

@router.post("/asr", response_model=ASRResult , name="ASR + diarization")
def audio_search(request: Request, audio:bytes=File(...), format: str = Header("mp3"), enable_align: bool = Header(True), enable_diarization: bool = Header(True)) -> ASRResult:
    engine: STT = request.app.state.asr
    message = ""
    s_time = time()
    try:
        audio = AudioSegment.from_file(BytesIO(audio), format=format)
        audio = normalize(audio)
        result = engine.apply(audio, enable_align=enable_align, enable_diarization=enable_diarization)
    except Exception as error:
        logging.error(traceback.format_exc())
        message = str(error)
        result = []
    e_time = time()
    if isinstance(result, str):
        return ASRResult(result=[], took=e_time - s_time, message=result)
    return ASRResult(result=result["segments"], took=e_time - s_time, message=message)
