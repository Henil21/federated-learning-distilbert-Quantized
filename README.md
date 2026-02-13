# Federated Fine-Tuning of DistilBERT for Question Answering

A privacy-preserving federated learning system that fine-tunes **DistilBERT** for extractive Question Answering (SQuAD) across multiple clients — without any raw data ever leaving the client.

---

## 📖 Overview

This project demonstrates **federated learning** applied to NLP: two independent clients each train on a private shard of the SQuAD dataset, then upload only their model weights to a central server. The server aggregates these weights via **FedAvg** (federated averaging) and distributes the improved global model back to clients. No training data is shared at any point.

After training, the final model is **quantized** and **exported** to ONNX to reduce its size by ~2× with minimal accuracy loss.

```
Client 1 (SQuAD first half)          Client 2 (SQuAD second half)
        │  local training                      │  local training
        │  INT16 quantized weights             │  INT16 quantized weights
        └──────────────┬───────────────────────┘
                       ▼
              Federated Server
              (FedAvg aggregation)
                       │
              Global Model Updated
```

---

## 🗂️ Repository Structure

```
├── Fedrated_Server.ipynb   # Central aggregation server (Flask + bore tunnel)
├── client01.ipynb          # Client 1 — trains on first half of SQuAD
└── client02.ipynb          # Client 2 — trains on second half of SQuAD
```

---

## ✨ Features

- **Federated Averaging (FedAvg)** — server waits for all clients, then averages weights
- **INT16 weight quantization** — weights are quantized before transmission, cutting payload size ~2×
- **Gzip compression** — further compresses the quantized payload before HTTP transfer
- **Public tunnel via `bore`** — server exposes a public URL from Google Colab (no static IP needed)
- **Post-training quantization** — Dynamic INT8 (PyTorch) and ONNX INT8 export for deployment
- **SQuAD evaluation** — Exact Match and F1 score computed on the SQuAD validation set

---

## 🛠️ Setup & Usage

### Prerequisites

All three notebooks are designed to run on **Google Colab** (free tier works; GPU recommended for clients).

### 1. Start the Federated Server

Open `Fedrated_Server.ipynb` in Colab and run all cells in order.

```
pip install flask flask-cors transformers datasets torch
```

The server will:
1. Load the base `distilbert-base-uncased` model
2. Start a Flask API on port `5000`
3. Expose a public URL via `bore` (printed in the output, e.g. `http://bore.pub:XXXXX`)

Copy the printed `SERVER_URL` — you will need it for the clients.

---

### 2. Run the Clients

Open `client01.ipynb` and `client02.ipynb` in **separate Colab sessions** (or separate tabs).

In each notebook, update the `SERVER_URL` variable at the top:

```python
SERVER_URL = "http://bore.pub:XXXXX"   # ← paste your server URL here
```

Then run all cells. Each client will:
1. Load their SQuAD data shard (Client 1 → first half, Client 2 → second half)
2. Download the current global weights from the server
3. Fine-tune locally for the configured number of epochs
4. Quantize and compress their weights
5. Upload to the server

The server aggregates once both clients have submitted.

---

### 3. Post-Training: Quantization & Export

The final cells of `Fedrated_Server.ipynb` handle model compression:

| Export Format        | Size      | Reduction |
|----------------------|-----------|-----------|
| Original float32     | ~253 MB   | 1.0×      |
| Dynamic INT8 (PyTorch)| ~132 MB  | ~1.9×     |
| ONNX float32         | ~255 MB   | ~1.0×     |
| ONNX INT8            | ~64 MB    | ~4.0×     |

All compressed model files are downloaded automatically via `google.colab.files`.

---

## ⚙️ Configuration

Key hyperparameters in the client notebooks:

| Parameter         | Default | Description                            |
|-------------------|---------|----------------------------------------|
| `EPOCHS_PER_ROUND`| `1`     | Local training epochs per FL round    |
| `MAX_ROUNDS`      | `1`     | Total federated rounds                 |
| `BATCH_SIZE`      | `8`     | Training batch size                    |
| `LR`              | `3e-5`  | AdamW learning rate                    |
| `MAX_LENGTH`      | `384`   | Max token length for tokenizer         |
| `DOC_STRIDE`      | `128`   | Stride for long-document splitting     |

Server-side configuration in `Fedrated_Server.ipynb`:

| Parameter      | Default | Description                          |
|----------------|---------|--------------------------------------|
| `num_clients`  | `2`     | Number of clients to wait for        |
| `max_rounds`   | `1`     | Rounds before saving the final model |

---

## 📡 API Endpoints

The server exposes three REST endpoints:

| Endpoint           | Method | Description                                      |
|--------------------|--------|--------------------------------------------------|
| `/get_weights`     | GET    | Returns current global model weights (quantized + compressed) |
| `/submit_weights`  | POST   | Accepts client weights; triggers aggregation when all clients have submitted |
| `/status`          | GET    | Returns current round, client count, and training status |

---

## 📊 Evaluation

After training, the server notebook evaluates both the float32 and INT8 models on 500 SQuAD validation examples:

```
Model                     Size       EM         F1
──────────────────────────────────────────────────────
Original float32          253.20 MB  XX.XX%     XX.XX%
Dynamic INT8              131.72 MB  XX.XX%     XX.XX%
──────────────────────────────────────────────────────
```

A quantization F1 drop of **< 2%** is considered acceptable for deployment.

---

## 🧱 Tech Stack

- **Model**: [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased) (DistilBertForQuestionAnswering)
- **Dataset**: [SQuAD v1.1](https://huggingface.co/datasets/squad)
- **Frameworks**: PyTorch, HuggingFace Transformers, Datasets
- **Server**: Flask + flask-cors
- **Tunneling**: [bore](https://github.com/ekzhang/bore)
- **Compression**: INT16 quantization + gzip
- **Export**: ONNX (opset 13), `onnxruntime` INT8 quantization
- **Platform**: Google Colab

---

## 📝 Notes

- The `bore` tunnel URL changes every time the server is restarted. Always copy the fresh URL into the client notebooks before running them.
- Both client notebooks must be running simultaneously for aggregation to trigger (the server waits for all `num_clients` to submit before averaging).
- The server sets `MAX_CONTENT_LENGTH` to 512 MB to accommodate large model weight payloads.
- Reproducibility is ensured with `RANDOM_SEED = 42`; each client seeds with `RANDOM_SEED + CLIENT_ID` for diversity.

---

## 📜 License

MIT
