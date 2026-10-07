"""
HTTP server. The same code serves:
  - the desktop app (pywebview opens a window on localhost),
  - the browser (just open the address),
  - in the future — a mobile client (this is already a ready-made API).

Jobs take a long time (minutes), so: POST creates a job, the frontend polls
the progress and downloads the files when done.

Public-deployment knobs (environment variables):
  TABFORGE_TOKEN          if set, every /api/* request must carry it
                          (X-API-Token header, or ?token= for downloads)
  TABFORGE_WORKERS        parallel pipeline workers (default 1)
  TABFORGE_MAX_UPLOAD_MB  upload size limit (default 200)
  TABFORGE_MAX_DURATION_S audio length limit, seconds (default 600)
  TABFORGE_JOB_TTL_S      keep finished jobs this long (default 7200)
  TABFORGE_MAX_JOBS       stored-jobs cap (default 20)
    TABFORGE_JOBS_DIR       persistent job storage (default ~/.cache/tabforge/jobs)
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..audio.keydetect import Key
from ..audio.transcribe import abort_separation
from ..core.fretboard import TUNINGS
from ..pipeline import (STAGES, AnalyzeResult, PipelineOptions, StemAnalysis,
                        apply_bulk_edit, apply_repin, export_reference,
                        run_analyze, run_analyze_midi, run_transcribe)

if getattr(sys, "frozen", False):
    # PyInstaller bundle: data files are unpacked next to the binary
    FRONTEND = Path(getattr(sys, "_MEIPASS")) / "frontend"
else:
    FRONTEND = Path(__file__).resolve().parent.parent.parent.parent / "frontend"
JOBS_STORAGE = Path(os.environ.get(
    "TABFORGE_JOBS_DIR", str(Path.home() / ".cache" / "tabforge" / "jobs")))
WORK_ROOT = JOBS_STORAGE / "work"
WORK_ROOT.mkdir(parents=True, exist_ok=True)

TOKEN = os.environ.get("TABFORGE_TOKEN", "")
WORKERS = int(os.environ.get("TABFORGE_WORKERS", "1"))
# a 4-minute WAV is ~45 MB — the old 30 MB default rejected normal
# uploads; public deployments can still tighten this via the env
MAX_UPLOAD_BYTES = int(float(os.environ.get("TABFORGE_MAX_UPLOAD_MB", "200")) * 1e6)
MAX_DURATION_S = float(os.environ.get("TABFORGE_MAX_DURATION_S", "600"))
JOB_TTL_S = float(os.environ.get("TABFORGE_JOB_TTL_S", "7200"))
MAX_JOBS = int(os.environ.get("TABFORGE_MAX_JOBS", "20"))
SEPARATOR = os.environ.get("TABFORGE_SEPARATOR", "demucs")

CLEANUP_INTERVAL_S = 300.0


class JobCancelled(Exception):
    """Raised from the progress hook when the user pressed Stop."""


@dataclass
class Job:
    id: str
    status: str = "queued"  # queued | running | analyzed | done | error | canceled
    stage: str = ""
    log: list[str] = field(default_factory=list)
    analysis: list[dict] = field(default_factory=list)
    results: list[dict] = field(default_factory=list)
    error: str = ""
    bpm: float = 0.0                  # detected tempo, known после analyze
    backing: str = ""                 # download URL of the backing track
    song: str = ""                    # URL of the multi-track project gp5
    dir: Path | None = None
    audio: Path | None = None         # the uploaded file
    title: str = "Track"              # display name from the upload
    analyzed: object | None = None    # pipeline.AnalyzeResult (server-side)
    opts: object | None = None        # PipelineOptions of the last transcribe
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    # refreshed by the touch middleware on ANY /api/jobs/{id}/* call:
    # an open session must never expire under the user (calibration
    # session 2 lost two hours of flags to the TTL sweeper)
    last_access: float = field(default_factory=time.time)
    persisted_access: float = field(default_factory=time.time, repr=False)
    lock: threading.Lock = field(default_factory=threading.Lock)
    cancel: threading.Event = field(default_factory=threading.Event)

    def public(self) -> dict:
        with self.lock:
            return {
                "id": self.id, "status": self.status, "stage": self.stage,
                "stages": list(STAGES), "log": list(self.log[-30:]),
                "analysis": list(self.analysis), "bpm": self.bpm,
                "results": list(self.results), "error": self.error,
                "backing": self.backing, "song": self.song,
            }


JOBS: dict[str, Job] = {}
POOL = ThreadPoolExecutor(max_workers=max(1, WORKERS))


def _serialize_analysis(analyzed: AnalyzeResult, job_dir: Path) -> dict:
    stems = {}
    for name, path in analyzed.stems.items():
        try:
            stems[name] = str(path.relative_to(job_dir))
        except ValueError:
            continue
    midi_source = None
    if analyzed.midi_source:
        try:
            midi_source = str(analyzed.midi_source.relative_to(job_dir))
        except ValueError:
            pass
    return {
        "stems": stems,
        "analysis": {name: asdict(item)
                     for name, item in analyzed.analysis.items()},
        "bpm": analyzed.bpm, "beats": analyzed.beats,
        "tempo_reliable": analyzed.tempo_reliable,
        "key": asdict(analyzed.key) if analyzed.key else None,
        "warnings": analyzed.warnings, "midi_source": midi_source,
        "solo": analyzed.solo, "meter": analyzed.meter,
        "meter_changes": analyzed.meter_changes,
    }


def save_job_to_disk(job: Job) -> None:
    """Atomically persist the job snapshot; its working files live beside it."""
    if not job.dir:
        return
    try:
        with job.lock:
            data = {
                "id": job.id, "status": job.status, "stage": job.stage,
                "log": job.log[-100:], "analysis": job.analysis,
                "results": job.results, "error": job.error,
                "bpm": job.bpm, "backing": job.backing,
                "song": job.song, "title": job.title,
                "created_at": job.created_at,
                "finished_at": job.finished_at,
                "last_access": job.last_access,
                "audio": (str(job.audio.relative_to(job.dir))
                          if job.audio else None),
                "analyzed": (_serialize_analysis(job.analyzed, job.dir)
                             if job.analyzed else None),
                "opts": asdict(job.opts) if job.opts else None,
            }
            JOBS_STORAGE.mkdir(parents=True, exist_ok=True)
            target = JOBS_STORAGE / f"{job.id}.json"
            temporary = JOBS_STORAGE / f".{job.id}.{uuid.uuid4().hex}.tmp"
            temporary.write_text(json.dumps(data), encoding="utf-8")
            temporary.replace(target)
            job.persisted_access = job.last_access
    except Exception:  # noqa: BLE001 — in-memory state remains usable
        try:
            temporary.unlink(missing_ok=True)
        except (UnboundLocalError, OSError):
            pass


def load_job_from_disk(job_id: str) -> Job | None:
    """Restore a job snapshot and its stable working directory."""
    if len(job_id) != 12 or any(c not in "0123456789abcdef" for c in job_id):
        return None
    path = JOBS_STORAGE / f"{job_id}.json"
    job_dir = WORK_ROOT / job_id
    if not path.is_file() or not job_dir.is_dir():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("id") != job_id:
            return None
        job = Job(id=job_id)
        job.dir = job_dir
        job.status = data.get("status", "error")
        job.stage = data.get("stage", "")
        job.log = data.get("log", [])
        job.analysis = data.get("analysis", [])
        job.results = data.get("results", [])
        job.error = data.get("error", "")
        job.bpm = data.get("bpm", 0.0)
        job.backing = data.get("backing", "")
        job.song = data.get("song", "")
        job.title = data.get("title", "Track")
        job.created_at = data.get("created_at", time.time())
        job.finished_at = data.get("finished_at")
        job.last_access = data.get("last_access", time.time())
        job.persisted_access = job.last_access

        def stored_path(relative):
            if not relative:
                return None
            candidate = (job_dir / relative).resolve()
            return candidate if job_dir.resolve() in candidate.parents else None

        job.audio = stored_path(data.get("audio"))
        analyzed_data = data.get("analyzed")
        if analyzed_data:
            key_data = analyzed_data.get("key")
            job.analyzed = AnalyzeResult(
                stems={name: stored_path(value)
                       for name, value in analyzed_data.get("stems", {}).items()
                       if stored_path(value)},
                analysis={name: StemAnalysis(**item)
                          for name, item in analyzed_data.get("analysis", {}).items()},
                bpm=analyzed_data.get("bpm", 0.0),
                beats=analyzed_data.get("beats", []),
                tempo_reliable=analyzed_data.get("tempo_reliable", False),
                key=Key(**key_data) if key_data else None,
                warnings=analyzed_data.get("warnings", []),
                midi_source=stored_path(analyzed_data.get("midi_source")),
                solo=analyzed_data.get("solo", False),
                meter=analyzed_data.get("meter", 4),
                meter_changes=analyzed_data.get("meter_changes", []),
            )
        opts_data = data.get("opts")
        if opts_data:
            job.opts = PipelineOptions(**{
                **opts_data, "stems": tuple(opts_data.get("stems", ()))})
        if job.status in ("queued", "running"):
            job.status = "error"
            stage = job.stage or "analysis"
            job.error = (
                f"Server restarted during {stage}; this job cannot resume. "
                "If the restart was unexpected, check service logs and "
                "memory limits. A restart during Demucs separation can be "
                "caused by memory pressure."
            )
            job.finished_at = time.time()
            save_job_to_disk(job)
        return job
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _get_job(job_id: str) -> Job | None:
    job = JOBS.get(job_id)
    if job is None:
        job = load_job_from_disk(job_id)
        if job is not None:
            JOBS[job_id] = job
    return job


def _restore_persisted_jobs() -> None:
    for path in JOBS_STORAGE.glob("[0-9a-f]" * 12 + ".json"):
        job = load_job_from_disk(path.stem)
        if job is not None:
            JOBS[job.id] = job

app = FastAPI(title="TabForge")


# ---------------------------------------------------------------------------
# Auth: when TABFORGE_TOKEN is set, every /api/* call must present it.
# Download links and alphaTab can't send headers, so ?token= also counts.
# ---------------------------------------------------------------------------

@app.middleware("http")
async def require_token(request: Request, call_next):
    if TOKEN and request.url.path.startswith("/api/"):
        supplied = (request.headers.get("x-api-token")
                    or request.query_params.get("token"))
        if supplied != TOKEN:
            return JSONResponse({"detail": "invalid or missing API token"},
                                status_code=401)
    return await call_next(request)


# ---------------------------------------------------------------------------
# Job lifecycle: finished jobs expire after JOB_TTL_S, the store is capped
# at MAX_JOBS, and a background thread sweeps periodically.
# ---------------------------------------------------------------------------

def _drop_job(job_id: str) -> None:
    job = JOBS.pop(job_id, None)
    if job and job.dir:
        shutil.rmtree(job.dir, ignore_errors=True)
    (JOBS_STORAGE / f"{job_id}.json").unlink(missing_ok=True)


def cleanup_jobs(now: float | None = None) -> int:
    """Remove finished jobs older than JOB_TTL_S. Returns how many."""
    now = now if now is not None else time.time()
    expired = []
    for job_id, job in list(JOBS.items()):
        with job.lock:
            done = job.status in ("done", "error", "canceled")
            stamp = job.finished_at
        with job.lock:
            stamp = max(stamp or 0, job.last_access)
        if done and stamp and now - stamp > JOB_TTL_S:
            expired.append(job_id)
    for job_id in expired:
        _drop_job(job_id)
    return len(expired)


def _evict_for_capacity() -> bool:
    """Make room for one more job. Finished jobs go first (oldest first);
    returns False when everything is still queued/running."""
    if len(JOBS) < MAX_JOBS:
        return True
    finished = sorted(
        (j for j in JOBS.values()
         if j.status in ("done", "error", "canceled")),
        key=lambda j: j.finished_at or j.created_at)
    if not finished:
        return False
    _drop_job(finished[0].id)
    return len(JOBS) < MAX_JOBS


def _cleanup_loop() -> None:
    while True:
        time.sleep(CLEANUP_INTERVAL_S)
        cleanup_jobs()


@app.middleware("http")
async def _touch_job_middleware(request, call_next):
    parts = request.url.path.split("/")
    if len(parts) > 3 and parts[1] == "api" and parts[2] == "jobs":
        job = _get_job(parts[3])
        if job:
            now = time.time()
            with job.lock:
                persist_touch = now - job.persisted_access >= 60.0
                job.last_access = now
            if persist_touch:
                save_job_to_disk(job)
    return await call_next(request)


@app.on_event("startup")
def _start_cleanup_thread() -> None:
    _restore_persisted_jobs()
    cleanup_jobs()
    threading.Thread(target=_cleanup_loop, daemon=True,
                     name="tabforge-job-cleanup").start()


# ---------------------------------------------------------------------------
# Upload validation
# ---------------------------------------------------------------------------

def probe_duration(path: Path) -> float:
    """Seconds of audio; raises when the file is not decodable audio."""
    import librosa

    return float(librosa.get_duration(path=str(path)))


def _sanitized_upload_name(filename: str | None) -> str:
    """Only the extension of the client-supplied name survives: the name
    itself lands in a filesystem path, so traversal characters must never
    reach it."""
    suffix = Path(filename or "").suffix.lower()
    if not (2 <= len(suffix) <= 8) or not suffix[1:].isalnum():
        suffix = ".bin"
    return f"upload{suffix}"


async def _save_upload(file: UploadFile, target: Path) -> None:
    size = 0
    with target.open("wb") as fh:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    413, f"File is too large (limit "
                         f"{MAX_UPLOAD_BYTES // 1_000_000} MB)")
            fh.write(chunk)
    if size == 0:
        raise HTTPException(422, "Empty upload")


# ---------------------------------------------------------------------------
# The worker
# ---------------------------------------------------------------------------

def _progress_fn(job: Job):
    def progress(stage: str, msg: str) -> None:
        # cooperative cancellation: the pipeline calls this at every
        # step, so a raise here unwinds the whole run cleanly
        if job.cancel.is_set():
            raise JobCancelled()
        with job.lock:
            job.stage = stage
            job.log.append(msg)
        save_job_to_disk(job)
    return progress


def _run_analyze(job: Job, audio: Path,
                 separator: str = SEPARATOR,
                 use_mt3: bool = True,
                 solo: bool = False) -> None:
    with job.lock:
        canceled = job.cancel.is_set()   # canceled while still queued
        if canceled:
            job.status = "canceled"
            job.finished_at = time.time()
        else:
            job.status = "running"
    save_job_to_disk(job)
    if canceled:
        return
    try:
        def _on_card(analysis: dict) -> None:
            # progressive cards (task 76): partial analysis reaches
            # the poll loop while the arbiter is still listening
            with job.lock:
                job.analysis = [a.to_dict() for a in analysis.values()]
            save_job_to_disk(job)

        analyzed = run_analyze(audio, job.dir / "out", _progress_fn(job),
                               cancel_token=job.id, separator=separator,
                               use_mt3=use_mt3, solo=solo,
                               on_card=_on_card)
        with job.lock:
            job.analyzed = analyzed
            job.bpm = analyzed.bpm
            job.analysis = [a.to_dict()
                            for a in analyzed.analysis.values()]
            job.status = "analyzed"
            job.stage = "analyze"
        save_job_to_disk(job)
    except Exception as e:  # noqa: BLE001 — shown to the user
        with job.lock:
            # a killed demucs also surfaces as an exception: any failure
            # after the user pressed Stop is a cancellation, not an error
            if job.cancel.is_set():
                job.status = "canceled"
            else:
                job.status = "error"
                job.error = str(e)
    finally:
        with job.lock:
            if job.status == "running":
                job.status = "error"
                job.error = job.error or "analysis crashed unexpectedly"
            if job.status in ("error", "canceled"):
                job.finished_at = time.time()
        save_job_to_disk(job)


def _run_analyze_midi_job(job: Job, midi: Path) -> None:
    """The MIDI drop path: instant analyze, same job lifecycle."""
    with job.lock:
        canceled = job.cancel.is_set()
        if canceled:
            job.status = "canceled"
            job.finished_at = time.time()
        else:
            job.status = "running"
    save_job_to_disk(job)
    if canceled:
        return
    try:
        analyzed = run_analyze_midi(midi, job.dir / "out",
                                    _progress_fn(job))
        with job.lock:
            job.analyzed = analyzed
            job.bpm = analyzed.bpm
            job.analysis = [a.to_dict()
                            for a in analyzed.analysis.values()]
            job.status = "analyzed"
            job.stage = "analyze"
        save_job_to_disk(job)
    except Exception as e:  # noqa: BLE001 — shown to the user
        with job.lock:
            job.status = "error"
            job.error = str(e)
    finally:
        with job.lock:
            if job.status == "running":
                job.status = "error"
                job.error = job.error or "MIDI analysis crashed"
            if job.status in ("error", "canceled"):
                job.finished_at = time.time()
        save_job_to_disk(job)


def _run_transcribe(job: Job, opts: PipelineOptions) -> None:
    with job.lock:
        job.status = "running"
        job.results = []
        job.backing = ""
    save_job_to_disk(job)
    try:
        results = run_transcribe(job.dir / "out", job.analyzed, opts,
                                 _progress_fn(job))
        with job.lock:
            job.results = [
                r.to_dict(lambda stem, p:
                          f"/api/jobs/{job.id}/files/{stem}/{p.name}")
                for r in results
            ]
            if (job.dir / "out" / "backing" / "backing.wav").is_file():
                job.backing = f"/api/jobs/{job.id}/files/backing/backing.wav"
            if (job.dir / "out" / "song" / "song.gp5").is_file():
                job.song = f"/api/jobs/{job.id}/files/song/song.gp5"
            job.status = "done"
            job.stage = "done"
        save_job_to_disk(job)
    except Exception as e:  # noqa: BLE001 — shown to the user
        with job.lock:
            if job.cancel.is_set() and job.analyzed is not None:
                # canceled mid-transcription: the separation is intact,
                # so drop back to the instrument picker instead of dying
                job.cancel.clear()
                job.status = "analyzed"
                job.stage = "analyze"
                job.log.append("transcription canceled")
            else:
                job.status = "error"
                job.error = str(e)
    finally:
        # Whatever escaped above (BaseException included), a job must
        # never be left in "running" — the frontend would poll forever.
        with job.lock:
            if job.status == "running":
                job.status = "error"
                job.error = job.error or "job crashed unexpectedly"
            if job.status != "analyzed":
                job.finished_at = time.time()
        save_job_to_disk(job)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.post("/api/jobs")
async def create_job(file: UploadFile,
                     separator: str = Form(SEPARATOR),
                     use_mt3: str = Form("1"),
                     solo: str = Form("0")) -> dict:
    """Step 1: upload + separation + analysis. The job stops at status
    'analyzed' with per-instrument facts; step 2 is POST .../transcribe.
    separator: 'demucs' (fast, default) or 'roformer' (BS-Roformer-SW —
    measurably cleaner stems, ~30x slower on CPU)."""
    if separator not in ("demucs", "roformer"):
        raise HTTPException(400, f"Unknown separator: {separator}")
    cleanup_jobs()
    if not _evict_for_capacity():
        raise HTTPException(
            429, "The server is at its job limit right now — try again "
                 "in a few minutes")

    job = Job(id=uuid.uuid4().hex[:12])
    job.dir = WORK_ROOT / job.id
    job.dir.mkdir(parents=True)

    audio = job.dir / _sanitized_upload_name(file.filename)
    from ..audio.midi_in import is_midi, midi_project_facts
    try:
        await _save_upload(file, audio)
        if is_midi(audio):
            # a dropped MIDI: notes at face value, no audio checks
            try:
                _bpm, _beats, duration = midi_project_facts(audio)
            except Exception:
                raise HTTPException(
                    422, "This does not look like a readable MIDI file")
        else:
            try:
                duration = probe_duration(audio)
            except HTTPException:
                raise
            except Exception:
                raise HTTPException(
                    422, "This does not look like a decodable audio file")
        if duration > MAX_DURATION_S:
            raise HTTPException(
                413, f"Audio is too long ({duration:.0f}s; limit "
                     f"{MAX_DURATION_S:.0f}s)")
    except HTTPException:
        shutil.rmtree(job.dir, ignore_errors=True)
        raise

    job.audio = audio
    # display title: the original name, defanged (it reaches zip entry
    # names and a filesystem path in the reference export)
    raw = Path(file.filename or "").stem
    safe = "".join(c for c in raw
                   if c.isalnum() or c in " ._-")[:60].strip()
    job.title = safe or "Track"
    JOBS[job.id] = job
    save_job_to_disk(job)
    if is_midi(audio):
        POOL.submit(_run_analyze_midi_job, job, audio)
    else:
        POOL.submit(_run_analyze, job, audio, separator,
                    use_mt3 not in ("0", "false", "off", ""),
                    solo not in ("0", "false", "off", ""))
    return {"id": job.id}


@app.post("/api/jobs/{job_id}/transcribe")
async def transcribe_job(job_id: str, selection: dict) -> dict:
    """Step 2: transcribe the selected stems from the CACHED separation.
    May be called again with a different selection — demucs never reruns."""
    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    with job.lock:
        status = job.status
    if status not in ("analyzed", "done"):
        raise HTTPException(
            409, f"Job is not ready to transcribe (status: {status})")

    stems = tuple(s for s in selection.get("stems", []) if isinstance(s, str))
    if not stems:
        raise HTTPException(400, "Pick at least one instrument")
    # a MIDI project has no wav stems — its cards are the truth
    known = (set(job.analyzed.stems)
             if job.analyzed.stems else
             {s for s, a in job.analyzed.analysis.items()
              if a.status != "absent"})
    unknown = [s for s in stems if s not in known]
    if unknown:
        raise HTTPException(400, f"Unknown stems: {', '.join(unknown)}")
    tuning = selection.get("tuning", "standard")
    if tuning not in TUNINGS:
        raise HTTPException(400, f"Unknown tuning: {tuning}")
    # rhythm precision: transcription onsets carry ±50 ms of noise, so
    # eighths are the steady default — sixteenths are opt-in detail
    subdivision = selection.get("subdivision", 2)
    if subdivision not in (2, 3, 4):
        raise HTTPException(400, "subdivision must be 2, 3, or 4")
    # per-stem role override: what the stem should be WRITTEN as
    tempo_scale = float(selection.get("tempo_scale", 1.0))
    if tempo_scale not in (0.5, 1.0, 2.0):
        raise HTTPException(400, "tempo_scale must be 0.5, 1, or 2")
    treat = selection.get("treat", {}) or {}
    if not isinstance(treat, dict):
        raise HTTPException(400, "treat must be an object")
    for k, v in treat.items():
        if k not in known:
            raise HTTPException(400, f"Unknown stem in treat: {k}")
        if v not in ("guitar", "piano", "vocals"):
            raise HTTPException(400, f"Unknown role in treat: {v}")

    opts = PipelineOptions(
        stems=stems,
        tuning=tuning,
        subdivision=subdivision,
        # meter detected at analyze (madmom votes 3 vs 4; 4 without
        # the optional install) — the score's time signature
        beats_per_measure=getattr(job.analyzed, "meter", 4),
        treat={str(k): str(v) for k, v in treat.items()},
        tempo_scale=tempo_scale,
        guitar_engine=(str(selection.get("guitar_engine", "auto"))
                       if selection.get("guitar_engine", "auto")
                       in ("auto", "bp", "muscriptor", "gaps")
                       else "auto"),
        with_chords=bool(selection.get("with_chords", True)),
        with_lyrics=bool(selection.get("with_lyrics", True)),
        lyrics_lang=(str(selection["lyrics_lang"])[:8]
                     if selection.get("lyrics_lang") else None),
    )
    job.opts = opts
    save_job_to_disk(job)
    POOL.submit(_run_transcribe, job, opts)
    return {"id": job.id}


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict:
    """Stop a running analyze or transcribe. Cancel during analyze ends
    the job; cancel during transcribe drops back to the instrument
    picker (the cached separation survives)."""
    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    with job.lock:
        if job.status not in ("queued", "running"):
            raise HTTPException(
                409, f"Nothing to cancel (status: {job.status})")
        job.cancel.set()
        job.log.append("stopping…")
    save_job_to_disk(job)
    # demucs won't reach a cooperative checkpoint for minutes — kill it
    abort_separation(job.id)
    return {"id": job.id, "status": "canceling"}


@app.post("/api/jobs/{job_id}/repin")
async def repin_note(job_id: str, req: dict) -> dict:
    """Note editor: pin a note to a string (string=null removes the pin)
    and re-run the fingering around it. Fast — pure math, no audio."""
    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != "done" or job.opts is None:
        raise HTTPException(409, "Transcribe first, then edit")
    part = req.get("part")
    try:
        # the UI sends the beat position in alphaTab's quarter ticks
        # (960/quarter); the editor addresses notes on the fine grid
        # (24 units per beat) that the adaptive score is written on
        tick = round(int(req["qticks"]) * 24 / 960)
        result = apply_repin(
            job.dir / "out", part,
            tick=tick, pitch=int(req["pitch"]),
            string=(None if req.get("string") is None
                    else int(req["string"])),
            shared=job.analyzed, opts=job.opts)
    except (KeyError, TypeError):
        raise HTTPException(400, "repin needs part, qticks, pitch, string")
    except ValueError as e:
        raise HTTPException(400, str(e))

    with job.lock:
        for r in job.results:
            if r.get("stem") == part and result["ascii"]:
                r["ascii"] = result["ascii"]
    save_job_to_disk(job)
    return {"prev": result["prev"], "song": job.song}


@app.post("/api/jobs/{job_id}/repin_group")
async def repin_group(job_id: str, req: dict) -> dict:
    """Group edit: pin every note of one pitch (optionally only inside
    a selected range, qticks) to a string in one stroke. Passing
    `restore` (note_index -> previous pin) is the undo path."""
    from ..pipeline import apply_repin_group

    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != "done" or job.opts is None:
        raise HTTPException(409, "Transcribe first, then edit")
    part = req.get("part")
    try:
        def fine(field):
            v = req.get(field)
            return None if v is None else round(int(v) * 24 / 960)
        result = apply_repin_group(
            job.dir / "out", part,
            shared=job.analyzed, opts=job.opts,
            pitch=(None if req.get("pitch") is None
                   else int(req["pitch"])),
            string=(None if req.get("string") is None
                    else int(req["string"])),
            from_tick=fine("from_qticks"), to_tick=fine("to_qticks"),
            restore=req.get("restore"))
    except (KeyError, TypeError):
        raise HTTPException(400, "repin_group needs part and pitch "
                                 "(or restore)")
    except ValueError as e:
        raise HTTPException(400, str(e))

    with job.lock:
        for r in job.results:
            if r.get("stem") == part and result["ascii"]:
                r["ascii"] = result["ascii"]
    save_job_to_disk(job)
    return {"count": result["count"], "prev_pins": result["prev_pins"],
            "song": job.song}


@app.post("/api/jobs/{job_id}/bulk_edit")
async def bulk_edit(job_id: str, req: dict) -> dict:
    """Mass editor op (task 55): every note of a part inside a
    drag-selected range — octave shift, delete, collapse octave
    doubles, or reassign to another part. Pure math, no audio."""
    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != "done" or job.opts is None:
        raise HTTPException(409, "Transcribe first, then edit")
    try:
        sub = job.opts.subdivision
        result = apply_bulk_edit(
            job.dir / "out", req["part"],
            start_tick=round(int(req["from_qticks"]) * sub / 960),
            end_tick=round(int(req["to_qticks"]) * sub / 960),
            op=req["op"],
            shared=job.analyzed, opts=job.opts,
            target_part=req.get("target"),
            # pitch filter (calibration case #3, the bar-39 cluster:
            # 40 must go DOWN while 48 goes UP in the same range —
            # a range-only op cannot split them)
            pitch=(int(req["pitch"])
                   if req.get("pitch") is not None else None))
    except (KeyError, TypeError):
        raise HTTPException(
            400, "bulk_edit needs part, op, from_qticks, to_qticks")
    except ValueError as e:
        raise HTTPException(400, str(e))
    with job.lock:
        for r in job.results:
            new_ascii = result["ascii"].get(r.get("stem"))
            if new_ascii:
                r["ascii"] = new_ascii
    save_job_to_disk(job)
    return {"count": result["count"], "song": job.song}


@app.get("/api/jobs/{job_id}/notes/{part}")
async def part_notes(job_id: str, part: str) -> dict:
    """Note positions + confidence for the Review-mode overlay."""
    from ..pipeline import part_note_meta

    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != "done" or job.opts is None:
        raise HTTPException(409, "Transcribe first")
    try:
        notes = part_note_meta(job.dir / "out", part,
                               job.analyzed, job.opts)
    except ValueError as e:
        raise HTTPException(404, str(e))
    # Review threshold (block-70 tails): adaptive quantile, not a
    # constant — with the spectral confidence live, a fixed 0.5 marked
    # 45-100% of a routed part's notes on golden. Review's job is to
    # direct attention, so it flags the WORST ~15% of the part, and
    # never more than the constant used to allow.
    confs = sorted(n["conf"] for n in notes if not n.get("dead"))
    threshold = 0.5
    if len(confs) >= 20:
        threshold = min(0.5, confs[int(len(confs) * 0.15)])
    return {"notes": notes, "threshold": round(threshold, 3)}


# ---------------------------------------------------------------------------
# SoundFont (task 80): the alphaTab default (sonivox) is not good
# enough to verify notes BY EAR. MuseScore_General.sf3 (MIT, ~40MB)
# lazy-downloads on first request — the model-weights pattern — and is
# served locally afterwards; the frontend falls back to the CDN
# sonivox until the download lands.
# ---------------------------------------------------------------------------
# GeneralUser GS — the ONE bank alphaTab 1.4's synth can actually
# play in full: measured in real WebKit drives, the synth skips
# Vorbis samples (sf3: "type 20 not supported" — silence with
# soundFontLoaded still firing) AND stereo samples (FluidR3: "type
# 4/2 not supported", 1310 skips). GeneralUser GS is designed mono
# (that is WHY it is 30MB) and its license permits redistribution.
SOUNDFONT_URL = ("https://raw.githubusercontent.com/mrbumpy409/"
                 "GeneralUser-GS/main/GeneralUser-GS.sf2")
SOUNDFONT_PATH = Path.home() / ".cache" / "tabforge" / "GeneralUser-GS.sf2"
_sf_lock = threading.Lock()
_sf_downloading = False


def _fetch_soundfont() -> None:
    global _sf_downloading
    tmp = SOUNDFONT_PATH.with_suffix(".part")
    try:
        import urllib.request
        SOUNDFONT_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SOUNDFONT_URL, tmp)
        if tmp.stat().st_size > 10_000_000:      # sanity: a real sf2
            tmp.replace(SOUNDFONT_PATH)
        else:
            tmp.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001 — playback falls back to the CDN
        tmp.unlink(missing_ok=True)
    finally:
        _sf_downloading = False


@app.get("/api/soundfont/info")
async def soundfont_info() -> dict:
    """Ready flag + kick the lazy download on first ask."""
    global _sf_downloading
    ready = SOUNDFONT_PATH.exists()
    if not ready:
        with _sf_lock:
            if not _sf_downloading:
                _sf_downloading = True
                threading.Thread(target=_fetch_soundfont,
                                 daemon=True).start()
    return {"ready": ready}


@app.get("/api/soundfont")
async def soundfont():
    if not SOUNDFONT_PATH.exists():
        raise HTTPException(404, "soundfont not downloaded yet")
    return FileResponse(SOUNDFONT_PATH,
                        media_type="application/octet-stream")


@app.post("/api/jobs/{job_id}/flags")
async def add_flag(job_id: str, flag: dict) -> dict:
    """Calibration flags (task 77): the user marks "врёт вот тут"
    during playback; flags land in out/flags.json, travel inside the
    project archive, and become numbered cases on the next session."""
    import json
    import time as _time

    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    text = str(flag.get("text", ""))[:500]
    entry = {
        "bar": int(flag.get("bar", 0)),
        "qticks": int(flag.get("qticks", 0)),
        "part": str(flag.get("part", ""))[:40],
        "text": text,
        "ts": _time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = job.dir / "out" / "flags.json"
    flags = json.loads(path.read_text()) if path.exists() else []
    flags.append(entry)
    payload = json.dumps(flags, ensure_ascii=False, indent=1)
    path.write_text(payload)
    # durable mirror: flags must survive job expiry and app death —
    # calibration session 2 lost its marks with the swept job dir
    try:
        name = job.audio.stem if job.audio else job_id
        mirror = Path.home() / ".cache" / "tabforge" / "flags"
        mirror.mkdir(parents=True, exist_ok=True)
        (mirror / f"{name}.json").write_text(payload)
    except OSError:
        pass
    return {"count": len(flags)}


@app.get("/api/jobs/{job_id}/flags")
async def list_flags(job_id: str) -> dict:
    import json

    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    path = job.dir / "out" / "flags.json"
    return {"flags": json.loads(path.read_text())
            if path.exists() else []}


@app.get("/api/jobs/{job_id}/chords")
async def chords(job_id: str) -> dict:
    """The chord line (task 58): spans with names, positions in
    alphaTab quarter-ticks and fret diagrams from the actual tab."""
    import json

    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    path = (job.dir / "out" / "chords.json") if job.dir else None
    if not path or not path.exists():
        return {"chords": []}
    return {"chords": json.loads(path.read_text())}


@app.get("/api/jobs/{job_id}/lyrics")
async def lyrics(job_id: str) -> dict:
    """Word-level synced lyrics (task 60)."""
    import json

    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    path = (job.dir / "out" / "lyrics.json") if job.dir else None
    if not path or not path.exists():
        return {"segments": [], "language": None}
    return json.loads(path.read_text())


@app.post("/api/jobs/{job_id}/lyrics")
async def toggle_lyrics_segment(job_id: str, req: dict) -> dict:
    """Hide/show one segment (Suno pseudo-words die in one click);
    the .lrc and the gp5 lyrics channel follow."""
    import json

    from ..pipeline import Grid, _rebuild_outputs, scale_beats
    from ..audio.lyrics import to_lrc

    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != "done" or job.opts is None:
        raise HTTPException(409, "Transcribe first")
    path = job.dir / "out" / "lyrics.json"
    if not path.exists():
        raise HTTPException(404, "No lyrics")
    data = json.loads(path.read_text())
    try:
        idx = int(req["index"])
        data["segments"][idx]["hidden"] = bool(req["hidden"])
    except (KeyError, ValueError, IndexError, TypeError):
        raise HTTPException(400, "toggle needs index and hidden")
    path.write_text(json.dumps(data))
    lrc = job.dir / "out" / "vocals" / "lyrics.lrc"
    if lrc.parent.exists():
        lrc.write_text(to_lrc(data))
    try:
        state = json.loads((job.dir / "out" / "parts.json").read_text())
        beats = (scale_beats(job.analyzed.beats, job.opts.tempo_scale)
                 if job.opts.tempo_scale != 1.0 else job.analyzed.beats)
        grid = (Grid(beats, subdivision=job.opts.subdivision)
                if len(beats) > 1 else None)
        _rebuild_outputs(job.dir / "out", state, set(),
                         job.analyzed, job.opts, grid)
    except Exception:  # noqa: BLE001
        pass
    return {"segments": data["segments"], "song": job.song}


@app.get("/api/jobs/{job_id}/sections")
async def sections(job_id: str) -> dict:
    """Song structure (task 59): auto-detected, human-renamable."""
    import json

    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    path = (job.dir / "out" / "sections.json") if job.dir else None
    if not path or not path.exists():
        return {"sections": []}
    return {"sections": json.loads(path.read_text())}


@app.post("/api/jobs/{job_id}/sections")
async def rename_section(job_id: str, req: dict) -> dict:
    """Rename one section (automation proposes, the human refines);
    the gp5 markers follow."""
    import json

    from ..pipeline import Grid, _rebuild_outputs, scale_beats

    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != "done" or job.opts is None:
        raise HTTPException(409, "Transcribe first")
    path = job.dir / "out" / "sections.json"
    if not path.exists():
        raise HTTPException(404, "No sections detected")
    secs = json.loads(path.read_text())
    try:
        idx = int(req["index"])
        label = str(req["label"]).strip()[:40]
        if not label:
            raise ValueError
        secs[idx]["label"] = label
    except (KeyError, ValueError, IndexError, TypeError):
        raise HTTPException(400, "rename needs index and a label")
    path.write_text(json.dumps(secs))
    try:
        state = json.loads((job.dir / "out" / "parts.json").read_text())
        beats = (scale_beats(job.analyzed.beats, job.opts.tempo_scale)
                 if job.opts.tempo_scale != 1.0 else job.analyzed.beats)
        grid = (Grid(beats, subdivision=job.opts.subdivision)
                if len(beats) > 1 else None)
        _rebuild_outputs(job.dir / "out", state, set(),
                         job.analyzed, job.opts, grid)
    except Exception:  # noqa: BLE001 — the json is updated regardless
        pass
    return {"sections": secs, "song": job.song}


@app.get("/api/jobs/{job_id}/reference")
async def reference_zip(job_id: str):
    """Export the CURRENT (post-edit) notes as per-instrument MIDI
    named like the golden corpus — the correction becomes ground
    truth for the eval stand."""
    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != "done":
        raise HTTPException(409, "Transcribe first")
    title = job.title
    try:
        zip_path = export_reference(job.dir / "out", title)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return FileResponse(zip_path, filename=f"{title} reference.zip",
                        media_type="application/zip")


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str) -> dict:
    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job.public()


@app.get("/api/jobs/{job_id}/files/{stem}/{name}")
async def job_file(job_id: str, stem: str, name: str) -> FileResponse:
    job = _get_job(job_id)
    if not job or not job.dir:
        raise HTTPException(404, "Job not found")
    path = (job.dir / "out" / stem / name).resolve()
    if not path.is_file() or job.dir.resolve() not in path.parents:
        raise HTTPException(404, "File not found")
    return FileResponse(path, filename=name)


# ---------------------------------------------------------------------------
# Project save/load: one .tabforge archive — everything except the audio.
# The note editor works from parts.json + the saved beat grid, so an
# imported project is fully editable with no stems on disk.
# ---------------------------------------------------------------------------

PROJECT_META = "tabforge-project.json"
PROJECT_FORMAT = 1


@app.get("/api/jobs/{job_id}/project")
async def export_project(job_id: str):
    import json
    import zipfile

    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    with job.lock:
        if job.status != "done" or job.analyzed is None:
            raise HTTPException(409, "Transcribe first, then save")
        results = [dict(r) for r in job.results]
    a, o = job.analyzed, job.opts
    meta = {
        "format": PROJECT_FORMAT,
        "name": job.audio.stem if job.audio else "project",
        "bpm": a.bpm, "beats": list(a.beats),
        "tempo_reliable": a.tempo_reliable,
        "key": ({"tonic": a.key.tonic, "minor": a.key.minor,
                 "correlation": a.key.correlation} if a.key else None),
        "opts": {"stems": list(o.stems), "tuning": o.tuning,
                 "subdivision": o.subdivision,
                 "beats_per_measure": o.beats_per_measure,
                 "treat": dict(o.treat),
                 "tempo_scale": o.tempo_scale},
        "results": results,
    }
    out = job.dir / "out"
    path = job.dir / "project.tabforge"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob("*")):
            if p.is_dir():
                continue
            rel = p.relative_to(out)
            if rel.parts[0] == "stems" or p.name.startswith("_analyze_"):
                continue                      # audio stays out: too heavy
            z.write(p, str(rel))
        z.writestr(PROJECT_META, json.dumps(meta))
    return FileResponse(path, filename=f"{meta['name']}.tabforge")


def _remap_file_urls(results: list[dict], job_id: str) -> list[dict]:
    """Saved download URLs carry the OLD job id — rebuild them."""
    out = []
    for r in results:
        r = dict(r)
        files = {}
        for ext, url in (r.get("files") or {}).items():
            tail = url.split("/files/", 1)
            if len(tail) == 2:
                files[ext] = f"/api/jobs/{job_id}/files/{tail[1]}"
        r["files"] = files
        out.append(r)
    return out


@app.post("/api/projects")
async def import_project(file: UploadFile) -> dict:
    """Open a saved .tabforge: the job comes back in 'done' state, score
    and note editor fully working — re-transcribing needs the audio."""
    import json
    import zipfile

    cleanup_jobs()
    if not _evict_for_capacity():
        raise HTTPException(429, "The server is at its job limit right now")

    job = Job(id=uuid.uuid4().hex[:12])
    job.dir = WORK_ROOT / job.id
    job.dir.mkdir(parents=True)
    archive = job.dir / "import.tabforge"
    try:
        await _save_upload(file, archive)
        out = job.dir / "out"
        with zipfile.ZipFile(archive) as z:
            names = z.namelist()
            if PROJECT_META not in names:
                raise HTTPException(422, "Not a TabForge project file")
            for member in names:
                target = (out / member).resolve()
                if not str(target).startswith(str(out.resolve())):
                    raise HTTPException(422, "Malformed project archive")
            z.extractall(out)
        meta = json.loads((out / PROJECT_META).read_text())
        if meta.get("format") != PROJECT_FORMAT:
            raise HTTPException(422, "Unsupported project version")

        k = meta.get("key")
        job.analyzed = AnalyzeResult(
            stems={}, analysis={}, bpm=float(meta["bpm"]),
            beats=[float(t) for t in meta["beats"]],
            tempo_reliable=bool(meta.get("tempo_reliable", True)),
            key=Key(k["tonic"], k["minor"], k["correlation"]) if k else None)
        mo = meta["opts"]
        job.opts = PipelineOptions(
            stems=tuple(mo["stems"]), tuning=mo["tuning"],
            subdivision=int(mo["subdivision"]),
            beats_per_measure=int(mo.get("beats_per_measure", 4)),
            treat=dict(mo.get("treat", {})),
            tempo_scale=float(mo.get("tempo_scale", 1.0)))
        job.results = _remap_file_urls(meta.get("results", []), job.id)
        if (out / "backing" / "backing.wav").is_file():
            job.backing = f"/api/jobs/{job.id}/files/backing/backing.wav"
        if (out / "song" / "song.gp5").is_file():
            job.song = f"/api/jobs/{job.id}/files/song/song.gp5"
        job.status = "done"
        job.stage = "done"
        job.finished_at = time.time()
        job.log.append(f"project '{meta.get('name', 'project')}' opened")
    except HTTPException:
        shutil.rmtree(job.dir, ignore_errors=True)
        raise
    except (zipfile.BadZipFile, KeyError, ValueError, TypeError) as e:
        shutil.rmtree(job.dir, ignore_errors=True)
        raise HTTPException(422, f"Broken project file ({e})")

    JOBS[job.id] = job
    return {"id": job.id}


@app.get("/api/tunings")
async def tunings() -> dict:
    return {"tunings": sorted(TUNINGS)}


@app.get("/api/limits")
async def limits() -> dict:
    """The UI checks the file size BEFORE uploading — rejecting a 45 MB
    wav after a full upload is a bad way to say no."""
    from ..audio.arbiter import find_mt3
    from ..audio.lyrics import available as lyrics_available
    return {"max_upload_mb": MAX_UPLOAD_BYTES // 1_000_000,
            "max_duration_s": MAX_DURATION_S,
            "mt3_available": find_mt3() is not None,
            "lyrics_available": lyrics_available()}


# the frontend goes last so it doesn't intercept /api/*
app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")
