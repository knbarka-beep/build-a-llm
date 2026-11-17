Build-A-LLM

Work for the "From internet text to an AI assistant" case (AI in Accounting and Business, SDU).

Files by step

Step 1 and 2: fineweb_sample.json, concat.py, fineweb_corpus.txt, fineweb_bits.txt, fineweb_bytes.txt, fineweb_tokens.txt
Step 3: model.py, train.py, sampling.py, sky_completions.py, corpus_completions.py, corpus_completions.txt, requirements.txt, results/ (training log, samples, loss plot, checkpoint)
Step 4: step4_base_vs_instruct.txt
Step 5: conversations.json
Step 6: step6_hallucinations_thinking_tools.txt, article_cotrugli.txt
Step 7: local_model.png (added when the run is done)

NOTES.md explains what was done and what I saw in each step.

Run the training step

pip install -r requirements.txt
py train.py
py sky_completions.py
py corpus_completions.py

All numbers in NOTES.md come from my own runs. sky_completions.py fails on purpose: the word " sky" is not in the small training vocabulary.
