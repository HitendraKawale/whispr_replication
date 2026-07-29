"""LibriSpeech loading, splitting and batching.

LibriSpeech's layout is `<split>/<speaker>/<chapter>/<speaker>-<chapter>-<utt>.flac`
with one `.trans.txt` per chapter holding `<utt_id> <UPPERCASE TRANSCRIPT>`.

The split here is **speaker-disjoint**: no speaker in the validation set appears
in training. That is deliberately the harder choice — a random utterance split
would let the model memorise each voice and report a flattering number that
tells us nothing about generalisation.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import Dataset

from whispr import audio, mel
from whispr.config import AudioConfig

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "data" / "LibriSpeech"


@dataclass(frozen=True)
class Utterance:
    utt_id: str
    path: str
    text: str
    duration: float
    speaker: str

    @property
    def chapter(self) -> str:
        return self.utt_id.split("-")[1]


def scan_split(root: Path | str, split: str = "dev-clean") -> list[Utterance]:
    """Walk the corpus and build the utterance index.

    Reads durations from the FLAC headers only — `sf.info` does not decode
    audio, so a full scan of dev-clean takes about a second.
    """
    split_dir = Path(root) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"{split_dir} not found. Fetch it with:\n"
            f"  uv run python scripts/04_data.py --download"
        )

    utterances: list[Utterance] = []
    for trans in sorted(split_dir.glob("*/*/*.trans.txt")):
        for line in trans.read_text().splitlines():
            if not line.strip():
                continue
            utt_id, text = line.strip().split(" ", 1)
            flac = trans.parent / f"{utt_id}.flac"
            if not flac.exists():
                continue
            utterances.append(
                Utterance(
                    utt_id=utt_id,
                    path=str(flac),
                    text=text,
                    duration=sf.info(str(flac)).frames / audio.SAMPLE_RATE,
                    speaker=utt_id.split("-")[0],
                )
            )
    return utterances


def cached_index(root: Path | str, split: str, cache: Path | str | None = None) -> list[Utterance]:
    """Scan once, then reuse. The index is derived data, so it lives in data/."""
    cache = Path(cache) if cache else Path(root).parent / f"index_{split}.json"
    if cache.exists():
        return [Utterance(**row) for row in json.loads(cache.read_text())]
    index = scan_split(root, split)
    cache.write_text(json.dumps([u.__dict__ for u in index], indent=0))
    return index


def speaker_split(
    utterances: list[Utterance],
    val_speakers: int = 6,
    seed: int = 1234,
) -> tuple[list[Utterance], list[Utterance]]:
    """Hold out whole speakers for validation.

    Speaker-disjoint rather than utterance-random, so validation measures
    generalisation to a new voice rather than memorisation of a familiar one.
    With 40 speakers, holding out 6 gives roughly a 15% validation set.
    """
    speakers = sorted({u.speaker for u in utterances})
    rng = random.Random(seed)
    held_out = set(rng.sample(speakers, val_speakers))
    train = [u for u in utterances if u.speaker not in held_out]
    val = [u for u in utterances if u.speaker in held_out]
    return train, val


class LibriSpeechDataset(Dataset):
    """Utterances as (log-mel, text) pairs.

    Utterances longer than the configured window are **dropped, not truncated**.
    Truncating audio while keeping the full transcript would train the model to
    invent the words it can no longer hear — a direct way to manufacture
    hallucination, which is exactly the failure mode the paper's §2.4 discusses.
    """

    def __init__(
        self,
        utterances: list[Utterance],
        config: AudioConfig | None = None,
        tokenizer=None,
        augment: bool = False,
    ) -> None:
        self.config = config or AudioConfig()
        self.tokenizer = tokenizer
        self.augment = augment

        self.utterances = [u for u in utterances if u.duration <= self.config.window_seconds]
        self.dropped = len(utterances) - len(self.utterances)

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, i: int) -> dict:
        utt = self.utterances[i]
        wav = audio.load_audio(utt.path, self.config.sample_rate)

        if self.augment:
            wav = _augment_waveform(wav)

        spec = mel.log_mel_spectrogram(
            wav,
            n_mels=self.config.n_mels,
            sample_rate=self.config.sample_rate,
            hop_length=self.config.hop_length,
            pad_to=self.config.n_samples,
        )

        item = {
            "mel": spec,
            "text": utt.text,
            "utt_id": utt.utt_id,
            "speaker": utt.speaker,
            "duration": utt.duration,
        }
        if self.tokenizer is not None:
            tokens = self.tokenizer.encode_training(utt.text)
            item["tokens"] = torch.tensor(tokens, dtype=torch.long)
        return item

    def total_hours(self) -> float:
        return sum(u.duration for u in self.utterances) / 3600


def _augment_waveform(wav: torch.Tensor) -> torch.Tensor:
    """Mild gain jitter.

    The paper uses *no* augmentation for the original models — with 680k hours
    they rely on data diversity instead (§2.4). We have 5 hours, so a little
    regularisation is warranted. Gain jitter specifically, because step 3
    showed the frontend is not loudness invariant: varying the level is
    teaching the model something the preprocessing genuinely does not provide.
    """
    gain = 10 ** (torch.empty(1).uniform_(-6, 6).item() / 20)
    return torch.clamp(wav * gain, -1.0, 1.0)


def collate(batch: list[dict], pad_token: int = 0) -> dict:
    """Stack a batch. Mels are already a fixed size; token sequences are not.

    Labels use -100 in the padded positions, which `cross_entropy` ignores, so
    padding contributes no gradient.
    """
    out = {
        "mel": torch.stack([b["mel"] for b in batch]),
        "text": [b["text"] for b in batch],
        "utt_id": [b["utt_id"] for b in batch],
    }

    if "tokens" in batch[0]:
        seqs = [b["tokens"] for b in batch]
        longest = max(len(s) for s in seqs)
        tokens = torch.full((len(seqs), longest), pad_token, dtype=torch.long)
        labels = torch.full((len(seqs), longest), -100, dtype=torch.long)
        for i, s in enumerate(seqs):
            tokens[i, : len(s)] = s
            labels[i, : len(s)] = s
        # Teacher forcing: predict token t+1 from tokens[:t].
        out["tokens"] = tokens[:, :-1].contiguous()
        out["labels"] = labels[:, 1:].contiguous()

    return out


def describe(utterances: list[Utterance]) -> str:
    """A one-glance summary, used by scripts and tests."""
    if not utterances:
        return "empty"
    durs = sorted(u.duration for u in utterances)
    total = sum(durs)
    words = sum(len(u.text.split()) for u in utterances)
    return (
        f"{len(utterances):,} utts · {total/3600:.2f} h · "
        f"{len({u.speaker for u in utterances})} speakers · "
        f"{words:,} words · "
        f"median {durs[len(durs)//2]:.1f}s · max {durs[-1]:.1f}s"
    )
