import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Make sure the repo root is importable when running from services/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
import json

import numpy as np
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from hwr.constants import DATA, PREPROCESS
from hwr.data.datarep import Point, PointSet
from hwr.decoding.ctc_decoder import BestPathDecoder, TrieBeamSearchDecoder
from hwr.models.ONHWRECO import ONHWRECO

model = None
decoder = None
CHAR2IDX = {c: i for i, c in enumerate(DATA.CHARS)}
BLANK_IDX = DATA.BLANK_IDX

# Verification margin (nats per timestep): if the target scores within this
# margin of the best decoded candidate, the writing counts as correct.
# Set BELOW the measured jab-vs-job separation (~0.29) so a clean "job" is
# never accepted as "jab" (and vice versa).
VERIFY_TOL = 0.15
# Near-miss: a candidate within 1 edit of the target that scored within this
# many nats/step of the best candidate counts as "very close / maybe a typo".
NEAR_TOL = 0.15

class PointIn(BaseModel):
    x: float
    y: float
    t: float


class StrokeIn(BaseModel):
    points: list[PointIn] = Field(..., min_length=1)


class CheckIn(BaseModel):
    target: str
    strokes: list[StrokeIn] = Field(..., min_length=1)
    # Optional list of ALL possible words (e.g. the current card's word list).
    # When provided, decoding is constrained to these words only — this turns
    # recognition into a spelling/verification checker and avoids nonsense.
    vocab: list[str] | None = None


class CheckOut(BaseModel):
    ok: bool
    recognized: str
    target: str
    confidence: float
    margin: float | None = None
    candidates: list[str] = []
    near: bool = False


class CollectOut(BaseModel):
    saved: bool
    sample_id: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, decoder

    decoder = TrieBeamSearchDecoder(
        beam_width=25, lm='sbo', ngram=5, prune=100, trie='100k', gamma=1,
    )

    # Dummy decoder so the default TrieBeamSearchDecoder(7gram) is never
    # constructed (it would crash: 7gram-p10.pkl is not present).
    model = ONHWRECO(preload=True, gru=False, decoder=BestPathDecoder())
    model.compile()

    # Warm up: run one tiny prediction so TF allocates its graph/GPU buffers
    # now, not on the first real user request.
    zeros = np.zeros((1, 10, 6), dtype='float32')
    model.pred_model.predict(zeros, verbose=0)
    print("Model + LM loaded. Ready.")

    yield

    # optional cleanup (close TF session etc.)


app = FastAPI(title="HWR API", lifespan=lifespan)

# Allow requests from the React dev server / app.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Build the (N, 6) feature matrix: exactly like reader.py + datarep.py do
# for IAM data: strokes -> points -> preprocess(SCHEME6) -> line features.
# ---------------------------------------------------------------------------
def strokes_to_pointset(strokes: list[StrokeIn]) -> PointSet:
    points = []
    min_t = min(p.t for s in strokes for p in s.points)
    for stroke_id, s in enumerate(strokes, start=1):
        for p in s.points:
            points.append(Point(stroke=stroke_id,
                                time=(p.t - min_t) * 1000,
                                x=p.x,
                                y=p.y))
    return PointSet(points=points)


def _confidence(sm: np.ndarray) -> float:
    # Simple confidence: mean of the winning class probability per timestep.
    if sm.ndim == 3:
        sm = sm[0]
    argmax = np.argmax(sm, axis=-1)
    return float(np.mean(sm[np.arange(sm.shape[0]),
                            argmax]))


def ctc_forward_logprob(sm: np.ndarray, text: str):
    """Standard CTC forward algorithm: log P(text | softmax).

    sm: (T, C); text: string label using the model's charset.
    Returns log probability (float) or None if any char is out of vocab.
    """
    idx = [CHAR2IDX[c] for c in text if c != '%']
    if len(idx) != len(text):
        return None
    T, _ = sm.shape
    L = len(idx)
    if L == 0:
        return 0.0
    pb = sm[:, BLANK_IDX]
    n_states = 2 * L + 1
    alpha = np.zeros((T, n_states))
    alpha[0, 0] = pb[0]
    alpha[0, 1] = sm[0, idx[0]]
    for t in range(1, T):
        prev, cur = alpha[t - 1], alpha[t]
        pt = sm[t]
        for s in range(n_states):
            if s % 2 == 0:
                cur[s] = pt[BLANK_IDX] * (prev[s] + (prev[s - 1] if s >= 1 else 0.0))
            else:
                emit = pt[idx[s // 2]]
                v = prev[s] + (prev[s - 1] if s >= 1 else 0.0)
                if s >= 3 and idx[s // 2] != idx[(s - 3) // 2]:
                    v += prev[s - 2]
                cur[s] = emit * v
    return float(np.log(alpha[T - 1, n_states - 2] + alpha[T - 1, n_states - 1]))


def _norm(s: str) -> str:
    return s.strip().lower()


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two strings (for spelling/typo checks)."""
    a, b = _norm(a), _norm(b)
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def _decode_with_trie(sm: np.ndarray, trie, top_n: int = 5):
    """Beam-search decode reusing the already-loaded LM, but with a custom trie."""
    if trie is None:
        return decoder.decode(rnn_out=[sm], top_n=top_n)[0]
    d = TrieBeamSearchDecoder(beam_width=decoder.beam_width, gamma=decoder.gamma)
    d.lm = decoder.lm
    d.ngram = decoder.ngram
    d.trie = trie
    return d.decode(rnn_out=[sm], top_n=top_n)[0]


def verify_score(sm: np.ndarray, target: str, vocab: list[str] | None = None):
    """How well does the writing match `target`.

    IMPORTANT: the verdict is always computed against the UNCONSTRAINED decode.
    If the beam is forced inside a small lesson trie, an off-vocab word like
    "hausen" gets snapped to the nearest lesson word ("house") and the writing
    is wrongly accepted. Free decode keeps the comparison honest.

    Returns (ok, recognized, margin_per_step, target_logp_per_step,
             best_logp_per_step, candidates, near).
    """
    T = sm.shape[0]
    # Free decode (100k trie) -> honest best candidate.
    cands = _decode_with_trie(sm, None, top_n=5)
    best, best_lp_ps = None, None
    for c in cands:
        clean = _norm(c)
        if not clean:
            continue
        lp = ctc_forward_logprob(sm, clean)
        if lp is None:
            continue
        lp_ps = lp / T
        if best_lp_ps is None or lp_ps > best_lp_ps:
            best, best_lp_ps = clean, lp_ps

    # Surface nearby vocab words as display candidates (no verdict effect).
    if vocab:
        for w in vocab:
            wn = _norm(w)
            if not wn or wn == best:
                continue
            lp = ctc_forward_logprob(sm, wn)
            if lp is None:
                continue
            lp_ps = lp / T
            if lp_ps >= (best_lp_ps if best_lp_ps is not None else 0.0) - NEAR_TOL:
                cands.append(w)

    target_n = _norm(target)
    tl = ctc_forward_logprob(sm, target_n)
    if tl is None:
        return False, (best or ""), None, None, None, cands, False
    t_ps = tl / T
    margin = t_ps - (best_lp_ps if best_lp_ps is not None else 0.0)

    # Near-miss: the target itself is not the winner, but it is within one edit
    # of the best candidate AND the best candidate isn't wildly better. This
    # forgives single-letter handwriting confusions (a<->o: jab vs job, etc.)
    # while still catching genuinely different words.
    near_edit = best is not None and _edit_distance(best, target_n) <= 1
    near = near_edit and best_lp_ps is not None and margin >= -NEAR_TOL
    exact = target_n == (best or "")

    ok = exact or margin >= -VERIFY_TOL or near
    return ok, (best or ""), margin, t_ps, best_lp_ps, cands, near


# ---------------------------------------------------------------------------
# Data collection: every /check call is recorded for later fine-tuning.
# ---------------------------------------------------------------------------
COLLECT_DIR = Path(__file__).resolve().parent.parent / "collected"
COLLECT_FILE = COLLECT_DIR / "samples.jsonl"


def save_sample(data: CheckIn, ok: bool, recognized: str, confidence: float,
                margin: float | None, cands: list[str]) -> str:
    sample_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    record = {
        "id": sample_id,
        "ts": datetime.now(timezone.utc).isoformat(),
        "target": data.target,
        "vocab": data.vocab,
        "ok": ok,
        "recognized": recognized,
        "confidence": confidence,
        "margin": margin,
        "candidates": cands,
        "strokes": [{"points": [{"x": p.x, "y": p.y, "t": p.t} for p in s.points]}
                    for s in data.strokes],
    }
    COLLECT_DIR.mkdir(parents=True, exist_ok=True)
    with open(COLLECT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return sample_id


# ---------------------------------------------------------------------------
# POST /check  ->  { "target": "...", "strokes": [...], "vocab": [...] } -> { ok, recognized }
# Verifies the writing against the known target word, without free recognition.
# ---------------------------------------------------------------------------
@app.post("/check", response_model=CheckOut)
async def check(data: CheckIn) -> CheckOut:
    ps = strokes_to_pointset(data.strokes)
    features = ps.generate_features(preprocess=PREPROCESS.SCHEME6)
    x = np.asarray([features], dtype='float32')
    sm = model.pred_model.predict(x, verbose=0)[0]

    ok, recognized, margin, t_ps, b_ps, cands, near = await run_in_threadpool(
        verify_score, sm, data.target, data.vocab)
    confidence = _confidence(sm)

    save_sample(data, ok, recognized, confidence, margin, cands)

    return CheckOut(
        ok=ok,
        recognized=recognized,
        target=data.target,
        confidence=confidence,
        margin=margin,
        candidates=cands,
        near=near,
    )


# ---------------------------------------------------------------------------
# POST /collect -> explicit version of /check that only stores the sample.
# ---------------------------------------------------------------------------
@app.post("/collect", response_model=CollectOut)
async def collect(data: CheckIn) -> CollectOut:
    ps = strokes_to_pointset(data.strokes)
    features = ps.generate_features(preprocess=PREPROCESS.SCHEME6)
    x = np.asarray([features], dtype='float32')
    sm = model.pred_model.predict(x, verbose=0)[0]

    ok, recognized, margin, t_ps, b_ps, cands, near = await run_in_threadpool(
        verify_score, sm, data.target, data.vocab)

    sample_id = save_sample(data, ok, recognized, _confidence(sm), margin, cands)
    return CollectOut(saved=True, sample_id=sample_id)


# ---------------------------------------------------------------------------
# POST /predict -> just recognize text, no matching.
# ---------------------------------------------------------------------------
@app.post("/predict")
async def predict(strokes: list[StrokeIn]) -> dict:
    recognized, confidence = await run_in_threadpool(recognize_from_strokes, strokes)
    return {"text": recognized, "confidence": confidence}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model_loaded": model is not None}