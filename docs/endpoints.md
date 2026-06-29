# Endpoint Contract

TAIScore uses two OpenAI-compatible chat completion endpoints.

## Refiner Endpoint

The refiner receives a prompt containing:

- the original user prompt,
- the initial actor response,
- the critic feedback.

It returns the full revised response. The default generation parameters in the reward implementation are:

```text
max_tokens = 2048
temperature = 0.2
top_p = 0.95
```

## Judge Endpoint

The judge receives the full transition:

```text
user prompt
initial response
critique
revised response
```

It returns JSON with integer scores from 1 to 10:

```json
{
  "critique_quality": {"score": 8, "reason": "..."},
  "critique_uptake": {"score": 8, "reason": "..."},
  "quality_gain": {"score": 7, "reason": "..."},
  "prompt_faithfulness": {"score": 9, "reason": "..."},
  "score": 8
}
```

The default judge parameters are:

```text
max_tokens = 2048
temperature = 0.2
top_p = 1.0
```

## URL Format

The client accepts any of these base URL forms:

```text
http://localhost:8000
http://localhost:8000/v1
http://localhost:8000/v1/chat/completions
```

Internally, all are normalized to a `/v1/chat/completions` request.

Use any serving stack that exposes an OpenAI-compatible chat completion API.
