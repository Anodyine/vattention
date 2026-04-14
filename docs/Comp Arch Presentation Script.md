**Intro Slide**
First, I'l briefly explain the impact of vAttention and the research questions, then Michel will go over the required terminology, Londy will cover the classical solutions and why they break down for KV cache growth, then I will come back and cover our analysis and experiment, and Josh will present the results and conclusions.

**Why use vAttention?**

"We’ll explain the technical details of this in more depth later, but for now the main idea is simple: vAttention is attractive because it is fast, but it can be harder to deploy.

The original vAttention paper reported up to 1.97 times faster token generation than PagedAttention variants. So there is a real performance reason to want to use vAttention for LLM serving.

The catch is that if you use the stock NVIDIA driver, you are limited to 2 megabyte pages, and those larger pages can still create tail waste. The vAttention project addressed that by implementing a custom driver path that reduces page size to 64 kilobytes.

But that custom driver adds real deployment complexity. It is harder to install, harder to maintain, and less practical than using the default driver stack. So our project asks when that extra complexity is actually necessary, and when 2 megabyte pages might already be good enough."

**Research Questions**

“That leads into our two main research questions: first, do we still need the custom vAttention driver that reduces page size from 2 megabytes to 64 kilobytes at long context lengths? And second, how can we predict before inference when 2 megabyte pages will cause meaningful tail waste for a given model?

**Michel will talk here
Then Londy**

**Attention Architectures**  
“For this project, the main thing to know is the relative trend across the three attention families we studied. Multi-head attention, or MHA, is good. Grouped-query attention, or GQA, is better. And multi-head latent attention, or MLA, is best in theory for reducing KV-cache memory.

We are not going to explain the internals of these architectures in detail today. The key point for our project is that these architectures store attention state differently, and that changes how many tokens fit into a 2 megabyte page. That difference is what shifts the fragmentation curve.

One novel part of this project is that MLA support did not already exist in the vAttention codebase we started from. We implemented that MLA path ourselves, and that made the MLA experiments in this presentation possible.

**Tail-Waste Expressions: Slide 1**  
“We next derived analytical expressions for tail waste. The exact expression, F exact of C, gives the true fragmentation percentage at a specific context length C. Because pages are allocated in fixed-size chunks, that exact curve has the sawtooth shape we expect.

We also use F worst of C, which is a smooth upper envelope. That gives us a simple way to estimate the peak fragmentation trend without plotting every page boundary.

On this slide, C is the context length in tokens, T is the number of tokens that fit in one 2 megabyte page, and the ceiling term tells us how many pages are mapped for that request. So the numerator is the unused capacity in the last page, and the denominator is the number of blocks that could be held by all mapped blocks”

**Tail-Waste Expressions: Slide 2**  
“The key architectural parameter is T, the number of tokens per page. For dense models like MHA and GQA, T depends on the page size, the number of KV heads on the local worker, the head dimension, and the bytes per stored element.

For MLA, the expression is different. Instead of dense key and value storage, T depends on the latent KV rank and the RoPE-related dimension that MLA keeps in the cache.

This slide is important because it shows that dense attention and MLA follow different formulas, but all of the inputs come from either context length or model parameters that are fixed during training. That means we can predict tail waste during inference directly from the model configuration, without first running the model experimentally.”

**Experiment**  
“To test this, we built an end-to-end pipeline around the serving system. First, the pipeline starts the model server. Then it sends multiple requests one at a time, using exact prompt token counts so we can control context length precisely. We keep max tokens equal to one so that we isolate allocator tail waste instead of mixing in more generation behavior and so that we aren't slowed down by the slow custom MLA decoding.

After the sweep finishes, the pipeline shuts the server down gracefully. On shutdown, the server automatically writes the metrics out to CSV. Then the final step reads those CSV files and generates the graphs.

So the workflow is: start server, run serial requests across many context lengths, shut down and save metrics, then plot the results. That gave us a clean and repeatable way to compare fragmentation behavior across architectures.”

**Synthetic MLA**  
“We also implemented synthetic MLA to make one particular comparison fair: GQA versus MLA on the same backbone. That gives us an apples-to-apples comparison, instead of comparing two totally different model families.

Here, synthetic does not mean fake math. The math and cache layout are real. It means we converted the model to behave like MLA for the memory analysis, but it is not a valid language model anymore. The runtime can execute, and the fragmentation measurements are meaningful, but the generated text would be nonsense. That was acceptable for this project because we care about allocator behavior, not output quality.”

**Analytical Graph / Threshold**  
“This analytical graph shows what we expected to see when we first proposed the project. For the Qwen 14B model, the fragmentation curve falls as context grows, and using our 5 percent threshold, the model would cross that line before 20,000 tokens.

The 5 percent threshold is arbitrary, but it gives us a simple reference point for comparing architectures and seeing how quickly fragmentation falls as context length increases.

That makes the motivation clear: for some models, 2 megabyte pages may become good enough at longer contexts. But as we’ll see in the results, that crossover point depends heavily on the attention architecture.

**Transition to Josh**  
“With that setup in place, Josh will now walk through the measured results and how well they matched the analytical predictions.”