"""StreamVoiceAnon+ extension around the official dual-AR model (no official file is modified or copied).

Training sequence (aivoice re-creation of the described method):

    [speaker condition (33)] [neutral prompt: sem_1 ac_1 ... sem_P ac_P] [SEP_sem SEP_ac] [emotional source ...]

* The prompt and SEP tokens are passed as the official ``spk_condition_len`` prefix, so the official forward
  computes the Slow-AR (semantic) and Fast-AR (codebook) losses on the *source* region only.
* SEP: two learned boundary embeddings, one for the semantic and one for the acoustic stream.
* Emotion distillation (training only): a forward hook on the Slow-AR output norm captures the hidden states
  at the source positions; a small head regresses frame-level emotion2vec features (MSE).
  L_total = L_slowAR + L_fastAR + lambda * L_emo. The head and teacher are dropped for inference; the
  inference graph is the official one (plus the two SEP embeddings in the prompt prefill).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..config import PlusConfig


class EmotionHead(nn.Module):
    def __init__(self, dim: int, emo_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, emo_dim))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class StreamVoiceAnonPlus(nn.Module):
    TRAINING_ONLY = ("emotion_head",)

    def __init__(self, arvc: nn.Module, cfg: PlusConfig) -> None:
        super().__init__()
        self.arvc, self.cfg = arvc, cfg
        dec = arvc.decoder                       # official DualARWrapper
        self.ar = dec.model                      # official DualARTransformer
        d = self.ar.config.dim
        self.sep = nn.Parameter(torch.randn(2, d) * 0.02)  # [semantic SEP, acoustic SEP]
        self.emotion_head = EmotionHead(d, cfg.emo_dim, cfg.emo_head_hidden)
        self._slow: torch.Tensor | None = None
        self.ar.norm.register_forward_hook(self._capture)

    def _capture(self, _m, _inp, out) -> None:
        self._slow = out

    def _delay(self) -> tuple[int, torch.Tensor, torch.Tensor]:
        dec = self.arvc.decoder
        d = dec.delay if isinstance(dec.delay, int) else dec.delay[int(torch.randint(0, len(dec.delay), (1,)))]
        D = self.ar.config.dim
        dev = self.sep.device
        w_end = dec.wait4end_embedding.weight[:d] if dec.wait4end_embedding is not None else torch.zeros(0, D, device=dev)
        w_start = dec.wait4start_embedding.weight[:d] if dec.wait4start_embedding is not None else torch.zeros(0, D, device=dev)
        return d, w_end, w_start

    @staticmethod
    def _interleave(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.stack([a, b], 1).reshape(-1, a.shape[-1])

    def forward(self, batch: dict[str, torch.Tensor], with_emotion: bool = True) -> dict[str, torch.Tensor]:
        """batch: sem_p [B,P], ac_p [B,K,P], sem_s [B,T], ac_s [B,K,T], src_lens [B], timbre [B,32,ctx], style [B,192],
        emo [B,T,emo_dim] (frame-level teacher features aligned to codec frames, training only)."""
        arvc, ar = self.arvc, self.ar
        spk = torch.cat([arvc.context_in(batch["timbre"]), arvc.style_in(batch["style"]).unsqueeze(1)], 1)
        cond_p, cond_s = arvc.embedding(batch["sem_p"]), arvc.embedding(batch["sem_s"])
        ac_p, ac_s = ar.embed(batch["ac_p"]), ar.embed(batch["ac_s"])
        delay, w_end, w_start = self._delay()
        seqs, lens = [], batch["src_lens"]
        for i in range(cond_s.shape[0]):
            sep = self.sep if self.cfg.use_sep else self.sep[:0]
            prefix = torch.cat([spk[i], self._interleave(cond_p[i], ac_p[i]), sep], 0)
            src = self._interleave(torch.cat([cond_s[i, :lens[i]], w_end], 0), torch.cat([w_start, ac_s[i, :lens[i]]], 0))
            seqs.append(torch.cat([prefix, src], 0))
        P = prefix.shape[0]
        emb = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
        pos = torch.arange(emb.shape[1], device=emb.device)[None].repeat(emb.shape[0], 1)
        out = ar(emb, batch["sem_s"], batch["ac_s"], lens, input_pos=pos, delay=delay, spk_condition_len=P)
        fast = F.cross_entropy(out.codebook_logits.transpose(1, 2), out.codebook_targets.long())
        slow = F.cross_entropy(out.token_logits.transpose(1, 2), out.token_targets.long(), ignore_index=-100)
        res = {"slow_ar": slow, "fast_ar": fast}
        total = self.cfg.slow_ar_weight * slow + self.cfg.fast_ar_weight * fast
        if with_emotion and "emo" in batch:
            h = self._slow  # [B, S, D] Slow-AR output (post-norm)
            idx = [torch.arange(delay * 2 + P, delay * 2 + P + int(lens[i]) * 2, 2, device=h.device) for i in range(h.shape[0])]
            pred = torch.cat([self.emotion_head(h[i, ix]) for i, ix in enumerate(idx)], 0)
            tgt = torch.cat([batch["emo"][i, : int(lens[i])] for i in range(h.shape[0])], 0)
            emo = F.mse_loss(pred, tgt.to(pred.dtype))
            res["emo"] = emo
            total = total + self.cfg.lambda_emo * emo
        res["loss"] = total
        return res

    def inference_state_dict(self) -> dict[str, torch.Tensor]:
        """Weights the official inference path needs: the ARVC model + the two SEP embeddings (no head)."""
        sd = {f"arvc.{k}": v for k, v in self.arvc.state_dict().items()}
        sd["sep"] = self.sep.detach()
        return sd
