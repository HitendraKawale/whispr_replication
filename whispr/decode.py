"""Decoding: turning audio into text, and scoring the result.

Greedy decoding uses an incremental KV cache, so generating a T-token
transcript costs O(T) decoder work instead of O(T^2). Measured on a 4-layer
decoder generating 120 tokens: 136 ms cached vs 283 ms uncached, with
identical output (asserted in tests/test_decode.py).

Beam search deliberately does *not* cache. Beams are re-ranked and pruned every
step, so the cache would have to be re-indexed by beam ancestry after each
prune — real bookkeeping that obscures what beam search actually is. Since beam
search here exists to explain the idea rather than to be fast, it recomputes the
prefix. Cross-attention keys and values are still computed once per utterance in
both paths, which is the larger saving anyway.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from whispr.model import Whispr
from whispr.tokenizer import WhisprTokenizer


@contextmanager
def _null_context():
    """Stand-in for `model.kv_cache()` when caching is disabled."""
    yield None


@dataclass
class DecodeResult:
    text: str
    tokens: list[int]
    avg_logprob: float


class Decoder:
    def __init__(
        self,
        model: Whispr,
        tokenizer: WhisprTokenizer,
        device: torch.device | None = None,
    ) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.max_len = model.config.n_text_ctx

    @torch.no_grad()
    def greedy(self, mel: torch.Tensor, use_cache: bool = True) -> list[DecodeResult]:
        """Always take the most likely next token.

        Fast and usually fine, but it cannot recover from a bad early choice:
        one wrong token early can force the rest of the sentence into
        nonsense, since the model conditions on its own mistake.

        With `use_cache=True` each step feeds only the newly generated token and
        reuses cached keys and values, making generation O(T) rather than O(T^2).
        `use_cache=False` recomputes the full prefix every step; the two produce
        the same tokens, which is what `tests/test_decode.py` asserts.
        """
        mel = self._prepare(mel)
        audio_features = self.model.encoder(mel)
        batch = mel.shape[0]

        tokens = torch.tensor(
            [self.tokenizer.sot_sequence()] * batch, device=self.device, dtype=torch.long
        )
        sum_logprobs = torch.zeros(batch, device=self.device)
        finished = torch.zeros(batch, dtype=torch.bool, device=self.device)
        # Per-sequence count, not a single step counter: in a batch where one
        # utterance finishes at 10 tokens and another runs to 90, dividing both
        # by 90 would report the short one as far less confident than it is.
        n_generated = torch.zeros(batch, device=self.device)
        steps = 0

        cache_ctx = self.model.kv_cache() if use_cache else _null_context()
        with cache_ctx as cache:
            while tokens.shape[1] < self.max_len and not finished.all():
                if cache is None:
                    logits = self.model.decoder(tokens, audio_features)[:, -1]
                else:
                    # First pass feeds the whole prompt; afterwards, one token.
                    step_in = tokens if steps == 0 else tokens[:, -1:]
                    offset = 0 if steps == 0 else tokens.shape[1] - 1
                    logits = self.model.decoder(
                        step_in, audio_features, kv_cache=cache, offset=offset
                    )[:, -1]

                logprobs = F.log_softmax(logits.float(), dim=-1)
                next_token = logprobs.argmax(dim=-1)

                # Once a sequence has emitted EOT, keep padding it so shapes align.
                next_token = torch.where(
                    finished, torch.full_like(next_token, self.tokenizer.special.eot), next_token
                )
                sum_logprobs += torch.where(
                    finished,
                    torch.zeros_like(sum_logprobs),
                    logprobs.gather(1, next_token[:, None])[:, 0],
                )
                n_generated += (~finished).float()
                steps += 1

                tokens = torch.cat([tokens, next_token[:, None]], dim=1)
                finished |= next_token == self.tokenizer.special.eot

        return self._finalise(tokens, sum_logprobs, n_generated)

    @torch.no_grad()
    def beam(self, mel: torch.Tensor, beam_size: int = 5, length_penalty: float = 1.0) -> list[DecodeResult]:
        """Keep the `beam_size` best partial sequences at every step.

        Greedy takes the best *token*; beam search approximates taking the best
        *sequence*. The distinction matters when a locally-unlikely token leads
        somewhere much better — which is common in speech, where the acoustic
        evidence for a word may only resolve a syllable later.

        Processes one utterance at a time for clarity; batching beams is an
        optimisation, not an idea.
        """
        mel = self._prepare(mel)
        results = []

        for i in range(mel.shape[0]):
            audio_features = self.model.encoder(mel[i : i + 1])
            beams = [(self.tokenizer.sot_sequence(), 0.0)]
            done: list[tuple[list[int], float]] = []

            while beams and len(beams[0][0]) < self.max_len:
                candidates = []
                tokens = torch.tensor([b[0] for b in beams], device=self.device, dtype=torch.long)
                feats = audio_features.expand(len(beams), -1, -1)

                logits = self.model.decoder(tokens, feats)[:, -1]
                logprobs = F.log_softmax(logits.float(), dim=-1)
                top = logprobs.topk(beam_size, dim=-1)

                for b, (seq, score) in enumerate(beams):
                    for k in range(beam_size):
                        token = int(top.indices[b, k])
                        new_score = score + float(top.values[b, k])
                        new_seq = seq + [token]
                        if token == self.tokenizer.special.eot:
                            done.append((new_seq, new_score))
                        else:
                            candidates.append((new_seq, new_score))

                if not candidates:
                    break
                # Rank by length-normalised score, or beam search systematically
                # prefers short sequences (every token adds negative logprob).
                candidates.sort(key=lambda x: x[1] / (len(x[0]) ** length_penalty), reverse=True)
                beams = candidates[:beam_size]

                if len(done) >= beam_size:
                    break

            pool = done or beams
            best_seq, best_score = max(
                pool, key=lambda x: x[1] / (len(x[0]) ** length_penalty)
            )
            n_gen = max(1, len(best_seq) - self.tokenizer.prompt_length)
            results.append(
                DecodeResult(
                    text=self._detokenise(best_seq),
                    tokens=best_seq,
                    avg_logprob=best_score / n_gen,
                )
            )
        return results

    # ------------------------------------------------------------------ utils

    def _prepare(self, mel: torch.Tensor) -> torch.Tensor:
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        return mel.to(self.device)

    def _detokenise(self, tokens: list[int]) -> str:
        content = [t for t in tokens if not self.tokenizer.is_special(t)]
        return self.tokenizer.decode(content).strip()

    def _finalise(self, tokens, sum_logprobs, n_generated) -> list[DecodeResult]:
        out = []
        counts = n_generated.tolist()
        for row, total, n in zip(tokens.tolist(), sum_logprobs.tolist(), counts):
            # Trim the EOT padding added to keep batch shapes aligned.
            eot = self.tokenizer.special.eot
            if eot in row:
                row = row[: row.index(eot) + 1]
            out.append(
                DecodeResult(
                    text=self._detokenise(row),
                    tokens=row,
                    avg_logprob=total / max(n, 1),
                )
            )
        return out


# --------------------------------------------------------------------- scoring

# Whisper's evaluation normalises text before scoring so that formatting
# differences don't count as recognition errors (paper §3.2, Appendix C). Our
# labels are already uppercase and unpunctuated, so the normaliser is short —
# but the principle matters, and WER can move by tens of points on it.
_NORMALISE_RE = re.compile(r"[^A-Z' ]+")


def normalise(text: str) -> str:
    """Uppercase, strip punctuation, collapse whitespace."""
    text = text.upper()
    text = _NORMALISE_RE.sub(" ", text)
    return " ".join(text.split())


def edit_distance(a: list[str], b: list[str]) -> int:
    """Levenshtein distance — the substitutions, insertions and deletions
    needed to turn `a` into `b`. Computed with a rolling row, O(min(n,m)) memory.
    """
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


def word_error_rate(references: list[str], hypotheses: list[str]) -> dict:
    """WER = (S + D + I) / N over the whole corpus.

    Aggregated over all utterances rather than averaged per-utterance: a
    one-word utterance getting one word wrong is 100% WER and would otherwise
    dominate the mean.
    """
    total_errors, total_words = 0, 0
    for ref, hyp in zip(references, hypotheses):
        r, h = normalise(ref).split(), normalise(hyp).split()
        total_errors += edit_distance(r, h)
        total_words += len(r)
    return {
        "wer": total_errors / max(total_words, 1),
        "errors": total_errors,
        "words": total_words,
    }
