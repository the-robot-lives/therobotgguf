"""robotgguf — retrofit existing LLM checkpoints onto the therobot runtime.

Stages (conversion-pipeline.md §2):
  R0 ingest   — survey the donor checkpoint                      [needs HF stack]
  R1 record   — forward-hook recordings of candidate cleave sites [needs HF stack]
  R2 cleave   — probe training + bottleneck selection             [numpy]
  R3 graft    — leaky state + modulator, function-preserving init [needs HF stack]
  R4 calibrate— delta thresholds from recording statistics        [numpy]
  R5 shims    — slice-scoped steering modules + admission         [numpy]
  R6 settle   — settle-track configuration                        [config only]
  R7 export   — extended GGUF emission + strip                    [gguf-py]
  R8 verify   — drives the fork's runtime binaries                [subprocess]

Prime directive: the donor core is frozen from the moment of recording; every
graft is function-preserving at insertion. A converted model with all
extensions at init is bit-for-bit the donor.
"""

SPEC_VERSION = 1

__all__ = ["SPEC_VERSION"]
