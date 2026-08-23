# Full AI Learning Roadmap

This is the long-term map. Detailed GitHub issues should normally be created only 1–2 weeks ahead so the daily plan can adapt.

## Phase 0 — Engineering & ML Environment
Learn Python for ML, NumPy, PyTorch tensors, CPU/GPU tensors, shapes, broadcasting, matrix multiplication, reproducible experiments, notebooks vs scripts, plotting and experiment logging.

Build: tensor playground, matrix-operation exercises, simple benchmark script.

Exit criteria:
- [ ] Comfortable manipulating tensors and shapes
- [ ] Can explain matrix multiplication in neural networks
- [ ] Can run and reproduce a PyTorch experiment

## Phase 1 — Mathematical Intuition for Neural Networks
Learn vectors/matrices, dot products, probability intuition, derivatives, partial derivatives, chain rule, gradients, mean/variance, logs/exponentials, softmax intuition.

Build: linear regression without autograd and a numerical derivative checker.

Experiments: learning-rate comparison; noisy vs clean data.

## Phase 2 — Neural Networks From First Principles
Learn neurons, weights/biases, activations, layers, MLPs, forward pass, losses, gradient descent, initialization, overfitting/underfitting, train/validation/test.

Build: single neuron, MLP, small classifier.

## Phase 3 — Backpropagation & Autograd
Learn computational graphs, local derivatives, chain rule through graphs, topological ordering, gradient accumulation, reverse-mode autodiff.

Build: scalar Value object, arithmetic ops, `.backward()`, tiny neural-network library.

Exit criteria: explain what `loss.backward()` does and match PyTorch gradients.

## Phase 4 — Language Modeling Foundations
Learn tokenization, vocabulary, character/word/subword approaches, BPE intuition, embeddings, logits, softmax, cross entropy, autoregressive modeling, temperature, top-k/top-p.

Build: tokenizer, bigram LM, text generation loop.

## Phase 5 — Attention & Transformers
Learn Q/K/V, scaled dot-product attention, causal masking, self-attention, multi-head attention, positional information, residuals, layer norm, feed-forward blocks, decoder-only transformers.

Build: self-attention, causal attention, MHA, transformer block, **TinyGPT**.

Experiments: remove mask, remove scaling, vary heads/embedding/context, visualize attention.

Exit criteria:
- [ ] Explain `softmax(QKᵀ / sqrt(d))V`
- [ ] Explain string → tokens → embeddings → transformer → logits → next token
- [ ] TinyGPT trains and generates text

## Phase 6 — Training LLMs
Learn datasets, batching, context windows, AdamW, LR schedules, warmup, weight decay, clipping, mixed precision, checkpointing, validation loss, perplexity, scaling, distributed-training concepts.

Build: repeatable TinyGPT training pipeline with checkpoints and metrics.

## Phase 7 — Open LLMs & Fine-Tuning
Learn Hugging Face, open-weight models, quantization, PEFT, LoRA, QLoRA, SFT, instruction tuning, preference-optimization concepts, and when to choose prompting vs RAG vs fine-tuning.

Build: fine-tuning pipeline, base-vs-tuned evaluation harness, quantized local inference.

## Phase 8 — Embeddings, Search & RAG
Learn embedding spaces, similarity, vector search, chunking, metadata filters, hybrid search, reranking, query rewriting, provenance and RAG failure modes.

Build: semantic search, baseline RAG, hybrid retrieval, reranking, RAG eval set.

Apply: Central Brain semantic memory/search.

## Phase 9 — Agents, Tools & MCP
Learn structured outputs, tool calling, agent loops, planner/executor patterns, state machines, retries, idempotency, human-in-the-loop, memory, MCP and multi-agent trade-offs.

Build: tool-using agent, stateful workflow, MCP server exposing safe Brain tools.

## Phase 10 — Evaluation, Reliability & Safety
Learn golden datasets, task metrics, rubric evaluation, LLM-as-judge limits, regression testing, prompt/model versioning, hallucination analysis, adversarial tests, guardrails, validation and tracing.

Build: eval harness, regression suite, failure taxonomy, model/prompt comparison.

## Phase 11 — LLM Inference & Performance
Learn prefill/decode, KV cache, TTFT, tokens/sec, batching, continuous batching, quantization, memory-bandwidth intuition, FlashAttention intuition, speculative decoding, vLLM, GPU constraints, routing, cost/latency/quality trade-offs.

Build: inference benchmark suite, local server, model router, KV-cache demo.

## Phase 12 — Speech & Voice AI
Learn waveforms, sample rate, spectrograms, mel features, speech recognition, TTS, VAD, turn detection, streaming, interruption handling and real-time latency.

Build: speech experiment, STT, TTS, real-time voice agent.

Apply: voice capture for Central Brain.

## Phase 13 — Image Generation
Learn image tensors, convolution intuition, autoencoders, VAEs, latent spaces, diffusion, noising/denoising, conditioning, latent diffusion, U-Net concepts and diffusion transformers.

Build: small autoencoder, tiny diffusion model, open image-model experiment.

## Phase 14 — Multimodal AI
Learn vision encoders, contrastive representations, image-text alignment, VLMs, multimodal tokenization, document understanding and multimodal RAG.

Build: image/document understanding, multimodal retrieval, multimodal assistant capability.

## Phase 15 — Recommendation & Personalization Systems
Learn ranking, candidate generation, explicit/implicit feedback, user/item representations, content-based systems, collaborative-filtering concepts, exploration/exploitation, temporal preferences, confidence-weighted observations and recommendation evaluation.

Build: personalized ranking prototype and explainable recommendations.

Apply: jobs, events, hobbies and learning recommendations.

## Phase 16 — Production AI Engineering
Learn FastAPI, async work, queues/workers, Postgres, Redis, object storage, observability, OpenTelemetry, prompt/model registries, secrets, auth, privacy, caching, rate limiting, retries, CI/CD, containers, cloud, cost and failure recovery.

Build: production architecture around flagship AI projects.

## Phase 17 — Senior AI Portfolio & Interviews
Portfolio evidence: AI From First Principles, Central Brain, a real-time voice or inference project, architecture docs, benchmarks, evaluations, failure retrospectives, demos and strong READMEs.

Interview prep: ML/LLM fundamentals, applied AI design, RAG, agents, evals, inference, experimentation, production trade-offs and project deep-dives.

Final exit criteria:
- [ ] Can explain model internals
- [ ] Can customize/fine-tune models
- [ ] Can build RAG/agent/multimodal systems
- [ ] Can evaluate them
- [ ] Can deploy and operate them
- [ ] Portfolio demonstrates senior-level engineering decisions
