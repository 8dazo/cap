# MPS Reference

Keep the tutorial honest about Apple Metal / MPS limitations.

## Guidance

- Assume single-device training on MPS.
- Prefer float16 autocast on Mac unless there is a proven bf16 path on the target setup.
- Mention CPU fallback when unsupported operations appear.
- Recommend small sanity runs before longer jobs.

## What to avoid

- Do not imply distributed MPS training support.
- Do not imply that tiny local models are comparable to large general-purpose assistants.
- Do not make Hacker News the default base dataset for the first successful run.
