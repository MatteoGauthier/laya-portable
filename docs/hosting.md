# Hosting the playground demo (Cloudflare Pages + R2)

The playground is a static Vite build, except for the model weights
(0.8–1.7GB per file). Cloudflare Pages caps files at 25MB, so the split is:

- **App shell** (`apps/playground/dist/`) → Cloudflare Pages (free, unlimited requests)
- **Weights** → R2 bucket `laya-models` (free tier covers this; zero egress fees)

## 0. Prerequisites

- `wrangler` logged in (`npx -y wrangler whoami`)
- AWS CLI (for >300MB uploads — wrangler CLI refuses files over 300 MiB)
- An R2 API token scoped to the bucket (Object Read & Write), used once via env vars, never committed

## 1. Push

```sh
git push -u origin main
```

## 2. Bucket (one time)

```sh
npx -y wrangler r2 bucket create laya-models
npx -y wrangler r2 bucket dev-url enable laya-models
# → https://pub-<hash>.r2.dev
```

## 3. Upload weights (S3 multipart — wrangler CLI caps at 300 MiB)

`wrangler r2 object put` refuses files over 300 MiB, so use the S3 API with
multipart. `aws s3 cp` works but proved fragile on flaky links (TLS
`BAD_RECORD_MAC` mid-stream); `rclone` with retries is the reliable path:

```sh
export RCLONE_CONFIG_R2_TYPE=s3 RCLONE_CONFIG_R2_PROVIDER=Cloudflare
export RCLONE_CONFIG_R2_ACCESS_KEY_ID=<key> RCLONE_CONFIG_R2_SECRET_ACCESS_KEY=<secret>
export RCLONE_CONFIG_R2_ENDPOINT=https://<account>.r2.cloudflarestorage.com RCLONE_CONFIG_R2_REGION=auto
rclone copyto models/laya-split-single.onnx R2:laya-models/laya-split-single.onnx \
  --s3-chunk-size 64M --s3-upload-concurrency 2 --transfers 1 \
  --retries 10 --retries-sleep 15s --timeout 15m --low-level-retries 20 --progress
rclone copyto models/laya-split-fp16.onnx R2:laya-models/laya-split-fp16.onnx # same flags
unset RCLONE_CONFIG_R2_ACCESS_KEY_ID RCLONE_CONFIG_R2_SECRET_ACCESS_KEY
```

Verify byte serving (expect 206 + `Content-Range`):

```sh
curl -s -o /dev/null -w "%{http_code}\n" -r 0-1023 https://pub-<hash>.r2.dev/laya-split-single.onnx
```

Multilingual / typed-decisions follow the same pattern (`.onnx` +
`.tokenizer.json` + `.rl_agent_config.json` + `.act_head.bin` + `.act_head.meta.json`
per task).

## 4. Public access + CORS (one time)

Public reads via the `r2.dev` URL from step 2. The worker `fetch()`es model
bytes cross-origin, so the bucket needs an open GET CORS policy — set it in
dash → R2 → bucket → CORS (the wrangler `cors set` JSON format fought back;
dashboard is reliable). Verify before deploying the app:

```sh
curl -s -D - -o /dev/null -H "Origin: https://<your-pages>.pages.dev" \
  https://pub-<hash>.r2.dev/<probe-file>
# expect: HTTP 200 + access-control-allow-origin: *
```

## 5. Build the app against R2

`apps/playground/src/worker.ts` reads model URLs from `VITE_MODELS_BASE_URL`
(default `/models`, which is what local dev serves). Bake in R2 at build time:

```sh
cd apps/playground
VITE_MODELS_BASE_URL=https://pub-<hash>.r2.dev npm run build
```

Confirm the URL landed in the bundle:

```sh
grep -rl "pub-<hash>" dist/assets/ | head -3
```

## 6. Deploy the shell

```sh
npx wrangler pages deploy dist --project-name=laya-playground
# → https://laya-playground.pages.dev
```

Custom domain (Pages project → Custom domains → `laya.ts.ax`, DNS handled
automatically on Cloudflare): the demo lives at `https://laya.ts.ax`.
No app change needed — models resolve via `VITE_MODELS_BASE_URL` and the
open R2 CORS policy already covers the new origin.

## 7. Verify live

1. Open the Pages URL, open DevTools → Network.
2. Run a prediction: expect the R2 download (waterfall `download` row),
   session init, then answers identical to local (billing 0.967 etc.).
3. Run again: `download 0ms` (Cache Storage), only inference remains.

## Costs & limits

- R2 free tier: 10GB storage, zero egress — this demo (~2.5GB English,
  ~5.5GB all tasks) fits.
- Pages free tier: unlimited requests for the static shell.
- To move off `*.r2.dev`: attach `models.<yourdomain>` as an R2 custom
  domain and rebuild with that base URL. Nothing else changes.
