# arXiv abstract classifier

A Django API that predicts arXiv subject codes from an abstract. It includes
synchronous single and batch endpoints, plus an asynchronous endpoint backed by
Celery and Redis.

## Quick demo

### 1. Download the data and train a small CPU demo model

Download and extract the arXiv metadata JSONL file with the system tools:

```bash
mkdir -p data
curl --fail --location --show-error \
  https://www.kaggle.com/api/v1/datasets/download/Cornell-University/arxiv \
  --output data/arxiv.zip
unzip -o data/arxiv.zip -d data
```

This downloads the public Kaggle arXiv metadata archive and extracts
`data/arxiv-metadata-oai-snapshot.json`. The extracted file is several gigabytes.
Skip these commands if you already have the dataset.

Build the images before running the training command:

```bash
docker compose build web celery-worker
```

Train a small CPU demo model:

```bash
docker compose run --rm --no-deps web python manage.py train_model \
  /app/data/arxiv-metadata-oai-snapshot.json \
  --max-records 100 \
  --max-per-label 3 \
  --preprocessing clean \
  --learning-rates 2e-5 \
  --epochs 1 \
  --batch-size 8 \
  --max-length 64
```

For this demo, `train_model.py` reads at most 100 lines from the metadata file,
skips papers without usable abstracts or subject codes, and limits how many
examples it keeps for each code. It then sets aside about one fifth of the
examples, fine-tunes the existing BERT model on the rest, and checks its
predictions against the held-back examples. Each paper keeps all its subject
codes, so the model can learn to return more than one code for an abstract.
The best run is saved under `classifier_api/model`, along with a report of the
evaluation. This tiny CPU run is just to make the demo work; its score is not a
meaningful measure of real-world accuracy.

The report is `classifier_api/model/training_report.json`. Its `dataset` section
summarizes what was read and kept, including counts by subject code;
`split` shows how many papers were used for training and validation. `labels`
lists the codes the model can predict. Each entry in `experiments` records a
preprocessing choice, learning rate, and validation metrics;
`selected_experiment` is the run with the best macro F1. In `metrics`,
precision reflects how many predicted codes were right, recall how many actual
codes were found, and F1 balances the two. With this small demo sample, use
these numbers to inspect the training flow, not as a measure of model quality.

The model is saved under `classifier_api/model`. If you are sending the project
without the model, the recipient can run this training step after downloading
the data.

### 2. Start the API, Redis, and worker

```bash
docker compose up -d --scale web=2
docker compose run --rm web python manage.py migrate
docker compose run --rm web python manage.py createsuperuser
```

The migration creates the user tables required for JWT login. Choose an email,
username, and password when prompted. Swagger is at <http://localhost/api/docs/>
and the health check is <http://localhost/api/health/>.
It reports Redis, Celery-worker, and model-load checks; it returns `503` if any
of them is unavailable.

### 3. Get an access token

Replace the placeholders with the account created above:

```bash
curl --fail-with-body --silent --show-error \
  -X POST http://localhost/api/auth/token/ \
  -H 'Content-Type: application/json' \
  -d '{"username":"YOUR_USERNAME","password":"YOUR_PASSWORD"}'
```

Copy the access and refresh values from the response. Send the access value
as `Bearer YOUR_ACCESS_TOKEN` in the `Authorization` header below. Access
tokens last 15 minutes; refresh tokens last one day.

### 4. Try the synchronous endpoints

Classify one abstract:

```bash
curl --fail-with-body --silent --show-error \
  -X POST http://localhost/api/classify/ \
  -H 'Authorization: Bearer YOUR_ACCESS_TOKEN' \
  -H 'Content-Type: application/json' \
  -d '{"abstract":"We study biological systems and cellular mechanisms in detail."}'
```

Example response:

```json
{
  "category": "q-bio.BM",
  "confidence": 0.87,
  "categories": [
    {"category": "q-bio.BM", "confidence": 0.87},
    {"category": "q-bio.CB", "confidence": 0.64}
  ],
  "cached": false
}
```

Classify a raw array of abstracts:

```bash
curl --fail-with-body --silent --show-error \
  -X POST http://localhost/api/classify/batch/ \
  -H 'Authorization: Bearer YOUR_ACCESS_TOKEN' \
  -H 'Content-Type: application/json' \
  -d '{"abstracts":["We study biological systems and cellular mechanisms in detail.","We analyze particle interactions and energy fields."]}'
```

Example response:

```json
{
  "predictions": [
    {
      "category": "q-bio.BM",
      "confidence": 0.87,
      "categories": [
        {"category": "q-bio.BM", "confidence": 0.87},
        {"category": "q-bio.CB", "confidence": 0.64}
      ],
      "cached": false
    },
    {
      "category": "physics.gen-ph",
      "confidence": 0.82,
      "categories": [
        {"category": "physics.gen-ph", "confidence": 0.82},
        {"category": "hep-ph", "confidence": 0.58}
      ],
      "cached": false
    }
  ]
}
```

The `categories` list shows ranked predictions, so an abstract can have more
than one predicted category. The category names and scores are examples; the
model's actual predictions will vary. A single prediction can also have
`"cached": true` when a recent matching abstract is found in Redis.

### 5. Refresh the access token

Send the `refresh` value from the token response:

```bash
curl --fail-with-body --silent --show-error \
  -X POST http://localhost/api/auth/token/refresh/ \
  -H 'Content-Type: application/json' \
  -d '{"refresh":"YOUR_REFRESH_TOKEN"}'
```

Use the returned `access` value for subsequent requests.

### 6. Try asynchronous prediction

Submit one abstract. The response includes a task ID:

```bash
curl --fail-with-body --silent --show-error \
  -X POST http://localhost/api/classify/async/ \
  -H 'X-Idempotency-Key: 550e8400-e29b-41d4-a716-446655440000' \
  -H 'Authorization: Bearer YOUR_ACCESS_TOKEN' \
  -H 'Content-Type: application/json' \
  -d '{"abstract":"We examine numerical methods for solving differential equations in complex systems."}'
```

The endpoint returns `202 Accepted` with a task ID:

```json
{
  "task_id": "38e481a5-e275-44b1-9968-f3b614a835fd",
  "status": "PENDING",
  "status_url": "http://localhost/api/classify/async/38e481a5-e275-44b1-9968-f3b614a835fd/"
}
```

If the abstract is already cached, the response still includes a task ID but
may already have `"status": "SUCCESS"`.

For safe retries, send a UUID in `X-Idempotency-Key`. Reuse it only when
retrying the same abstract; the API returns the original task ID. A different
abstract with that key returns `409 Conflict`. Keys expire after one hour.

If two requests arrive at nearly the same time with the same key and abstract,
only one task is queued. Both requests receive the same task ID:

```text
Request 1: X-Idempotency-Key: 550e8400-e29b-41d4-a716-446655440000
Response:  task_id: 38e481a5-e275-44b1-9968-f3b614a835fd

Request 2: X-Idempotency-Key: 550e8400-e29b-41d4-a716-446655440000
Response:  task_id: 38e481a5-e275-44b1-9968-f3b614a835fd
```

Use the returned ID to poll for the result:

```bash
curl --fail-with-body --silent --show-error \
  http://localhost/api/classify/async/YOUR_TASK_ID/ \
  -H 'Authorization: Bearer YOUR_ACCESS_TOKEN'
```

While the task is queued or running, polling returns `202 Accepted`:

```json
{
  "task_id": "38e481a5-e275-44b1-9968-f3b614a835fd",
  "status": "PENDING"
}
```

When it finishes, polling returns `200 OK` with the prediction under `result`:

```json
{
  "task_id": "38e481a5-e275-44b1-9968-f3b614a835fd",
  "status": "SUCCESS",
  "result": {
    "category": "q-bio.BM",
    "confidence": 0.87,
    "categories": [
      {"category": "q-bio.BM", "confidence": 0.87}
    ],
    "cached": false
  }
}
```

Only the user who submitted a task can view its result. Poll again if the
response status is still `PENDING` or `STARTED`.

## A few implementation notes

- Abstracts must be 20–20,000 characters. The synchronous single endpoint
  checks Redis before inference. The async endpoint checks the same cache:
  misses go through Celery batching, while hits get a task ID whose result is
  already complete for polling. Fresh async predictions are cached for later
  requests. Responses include the ranked subject-code list and confidence.
- JWTs are required for prediction and task-status routes. Besides identifying
  the user, that gives the rate limiter a stable per-user key. The DRF throttle
  uses Redis-backed rate limits across API processes, with distinct scopes for
  the prediction and polling endpoints.
- Each web worker runs blocking inference in its own single-thread executor.
  This keeps PyTorch work off the async event loop. We use one inference thread
  per process and bound the waiting work, so requests don't build an unbounded
  queue or compete through multiple simultaneous model calls.
- The async endpoint returns a task ID instead of making the client wait for
  model inference; the client can poll for its result. Celery Batches briefly
  groups independently submitted abstracts into one model call, reducing
  per-call overhead while keeping each prediction tied to its own task ID.
  Batches flush at `ASYNC_PREDICTION_BATCH_SIZE` or after
  `ASYNC_PREDICTION_BATCH_FLUSH_INTERVAL` seconds, whichever comes first. The
  defaults are 16 items and 0.03 seconds, and results expire after one hour.
- For separate single-abstract requests, prefer the async endpoint when the
  caller can accept polling: batching can amortize model-call overhead and
  improve throughput under concurrent load. It is not inherently more
  reliable, though; it adds queueing delay and depends on the broker and worker.
  Use the synchronous batch endpoint when the client already has a bulk request
  and wants one response, and the synchronous single endpoint for a simple
  request-response flow.
- The two paths can be scaled separately: add `celery-worker` replicas for
  async queue processing, or `web` replicas for more synchronous API capacity.
  Each replica adds model-serving processes and consumes CPU and memory, so
  scale against the machine's resources and measure throughput rather than
  assuming that more workers always make inference faster.

## Run tests


```bash
docker compose run --rm --no-deps web python manage.py test classifier_api.tests
```
