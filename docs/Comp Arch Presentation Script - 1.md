**Intro Slide**
Good evening, our project is about understanding memory fragmentation under vAttention, which is a method of saving the KV cache during large language model inference. To go over it, first I'll explain the impact of vAttention and the research questions we addressed, then Michel will go over the required terminology, Londy will cover the classical solutions and why they don't fit the KV cache application, then I'll come back and cover our analysis and experiment, and Josh will present the results and conclusions.

**Why use vAttention?**

"So why would anyone care about vAttention? We’ll explain the technical details of this in more depth later, but for now the main idea is that vAttention is attractive because it performs LLM inference faster than the alternatives, but it can be harder to deploy.

The original vAttention paper reported up to 1.97 times faster token generation than standard open source LLM inference software. So there is a real performance reason to use vAttention.

The catch is that if you use the stock NVIDIA driver, you're limited to large 2 megabyte pages, and those larger pages can cause memory to be wasted because of fragmentation. The vAttention project addressed that by implementing a custom driver that reduces  the page size to 64 kilobytes.

However, that custom driver adds real deployment complexity. It's harder to install, harder to maintain, and less practical than using the default driver stack. So our project asks when that extra complexity is actually necessary, and when 2 megabyte pages might already be good enough."

**Research Questions**

“That leads into our two main research questions: first, do we still need the custom vAttention driver at long context lengths? And second, how can we predict before inference when two megabyte pages will cause meaningful fragmentation for a given model?

**Michel will talk here
Then Londy**

**Attention Architectures**  
“We're not gonna to explain the internals of these architectures in detail today. The key point for our project is that these architectures store the KV cache differently, and that changes how many tokens fit into a each page. MHA uses more memory, GQA uses less, and, in theory, MLA uses even less.

One novel part of this project is that MLA support didn't already exist in the vAttention codebase we started with. We implemented the MLA path ourselves, which made the MLA experiments possible.

**Tail-Waste Expressions: Slide 1**  
“We also derived analytical expressions for fragmentation caused by the page size. F_exact of C, gives the true fragmentation percentage at a specific number of tokens C. Because each 2 MB page is allocated as a chunk, just after a page is allocated, but before it is filled, there should be a jump in the fragmentation. Therefore, we expected to see a sawtooth shape.

I'll quickly walk through this expression. T is the number of tokens that fit in one 2 megabyte page, given a specific model. Since the number of pages mapped must be an integer, the ceiling of C/T tells us how many pages would be mapped for a request of size C. Multiplying that by T, we can get the total number of tokens that could be held by all currently mapped pages, the mapped capacity. Then we can get the unused capacity in the last page by subtracting C from that quantity, divide by the mapped capacity, then multiply by 100 to get the percentage of memory that is wasted due to tail waste.

We also use F worst of C, which is a smooth upper envelope. That gives us a simple way to estimate the worst case scenario for fragmentation caused by the 2 megabyte page size."

**Tail-Waste Expressions: Slide 2**  
“Here is how we get T for the different types of models. The top expression here is for dense models like MHA and GQA and the bottom expression is for MLA models. 

We can't get into each parameter for this one, because of time, but this shows that even though dense attention and MLA follow different formulas, all of the terms in these expression are either page size or model parameters that are fixed at training time. That means we can predict tail waste before inference directly from the model configuration, system configuration, and context length without first running the model experimentally.”

**Analytical Graph / Threshold**  
“This analytical graph shows what we expected to see when we first proposed the project. For the Qwen 14B model, the fragmentation curve falls as context grows, and using our 5 percent threshold, the model would cross that line before 20,000 tokens.

The 5 percent threshold is arbitrary, but it gives us a simple reference point for comparing architectures and seeing how quickly fragmentation falls as context length increases.

That makes the motivation clear: for some models, 2 megabyte pages may become good enough at longer contexts. But as we’ll see in the results, that crossover point depends heavily on the specific model tested.

**Experiment**  
“To test this, we built an end-to-end pipeline around our modified vAttention system. First, the pipeline starts the model server. Then it sends requests one at a time, using fixed prompt token counts so we can control context length precisely. We set the parameter that controls the number of generated tokens to one.

After the sweep finishes, the pipeline shuts the server down gracefully. On shutdown, the server automatically writes the metrics out to CSV. Then the final step reads those CSV files and generates the graphs.”

**Synthetic MLA**  
“We also implemented synthetic MLA by making a Mistral GQA model to behave like an MLA model when it saves the KV cache. That allowed us to make the comparison between GQA and MLA on top identical model sizes, which gave us an apples-to-apples comparison between those architectures.

With synthetic MLA the cache layout is real. However changing the KV cache functionality without retraining causes it not to work as a language model. The program can execute, and the fragmentation measurements are meaningful, but the generated text is nonsense. That was acceptable for this project because we care about KV cache storage behavior, not output quality.”

**Transition to Josh**  
“Josh will now walk through the measured results and how well they matched the analytical predictions.”