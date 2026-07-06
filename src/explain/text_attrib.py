"""
APPROXIMATE explanation segment: word-level occlusion attributions for the
personality inference. For each word in the persona's text, mask it and
measure the change in each axis probability; a large drop means the word
supported that axis. This is the only non-exact link in the trace, and the
only one the deletion/insertion faithfulness tests target.
"""
from __future__ import annotations

import re

import numpy as np
import torch
import torch.nn.functional as F


class MBTIExplainer:
    """Wraps the fine-tuned classifier for embeddings, traits, occlusion."""

    def __init__(self, device: str = "cpu"):
        from src.config.settings import get_config, CHECKPOINT_DIR
        from src.models.bert_mbti.model import MBTIMultiLabelClassifier
        from transformers import AutoTokenizer

        self.config = get_config().bert
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = MBTIMultiLabelClassifier(config=self.config)
        ckpt = torch.load(CHECKPOINT_DIR / "bert_mbti" / "best_model.pt",
                          map_location=device, weights_only=False)
        self.model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        self.model.to(device).eval()
        self.device = device
        self.dims = self.model.DIMENSIONS  # ["EI","SN","TF","JP"]

    @torch.no_grad()
    def _forward(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """-> (cls [n,768] L2-normalised, trait probs [n,4])."""
        enc = self.tokenizer(texts, max_length=self.config.max_length,
                             padding="max_length", truncation=True,
                             return_tensors="pt").to(self.device)
        out = self.model.bert(input_ids=enc["input_ids"],
                              attention_mask=enc["attention_mask"])
        cls = out.last_hidden_state[:, 0, :]
        pooled = self.model.dropout(cls)
        probs = torch.cat(
            [F.softmax(self.model.classifiers[d](pooled), dim=-1)[:, 1:2]
             for d in self.dims], dim=1)
        return (F.normalize(cls, p=2, dim=-1).cpu().numpy(),
                probs.cpu().numpy())

    def embed(self, text: str, mbti_center: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray]:
        """Persona embedding (centered+renormalised, matching the
        precomputed users) and trait probabilities."""
        cls, probs = self._forward([text])
        m = cls[0] - mbti_center
        m /= np.linalg.norm(m) + 1e-8
        return m.astype(np.float32), probs[0]

    def occlusion(self, text: str, batch_size: int = 32
                  ) -> list[dict]:
        """
        Word-level occlusion. Returns one record per word:
        {word, deltas: {EI: dP(I), SN: dP(N), TF: dP(F), JP: dP(P)}}
        where delta = P(full text) - P(text without the word); positive
        delta means the word pushed the probability up.
        """
        words = re.findall(r"\S+", text)
        _, base = self._forward([text])
        variants = [" ".join(words[:i] + words[i + 1:])
                    for i in range(len(words))]
        probs = []
        for i in range(0, len(variants), batch_size):
            _, p = self._forward(variants[i:i + batch_size])
            probs.append(p)
        probs = np.vstack(probs)
        deltas = base[0][None, :] - probs      # [n_words, 4]
        return [{"word": w,
                 "deltas": {d: float(deltas[i, j])
                            for j, d in enumerate(self.dims)}}
                for i, w in enumerate(words)]
