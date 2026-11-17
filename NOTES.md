Build-A-LLM notes

Step 1: Training data

The FineWeb topic map shows how wide the training data is. It covers almost anything people write about online. I noticed topics like firearms, gaming and gambling, which I wouldn't expect a company to want its assistant to learn from.

My 10 documents are a random mix: a soap opera forum post, an angry blog rant, cancer news headlines, a recycling blog, a grocery industry article, a college running report, a Kraft product launch, a restaurant health inspection, an artist's news page and a page that only says "You must be a registered member to view this page".

So the data is not a clean library. It is whatever was on the web, login walls and rants included. Filtering removes a lot, but not everything.

All 10 documents are from the same Common Crawl snapshot (CC-MAIN-2013-20), collected in May 2013. A model only knows the world as it looked when its data was collected.

Step 2: Text representation

The first sentence is 133 characters, 133 bytes and 1 064 bits. It is plain ASCII, so one byte per character.

The whole corpus is 22 044 characters, 22 141 bytes and 4 731 tokens with the GPT-4o tokenizer. That is about 4,7 characters per token. There are more bytes than characters because 52 characters are not plain ASCII (curly apostrophes, ®) and take 2 to 3 bytes each in UTF-8.

In Tiktokenizer, common words like "the", "February" and "Viewing" are one token each, usually with the space in front. Rare words get split. "Marlena" was split into pieces starting with "Marl", and the "th" in "11th" is its own token. The colours only show where one token ends and the next starts.

This is why models are bad at counting letters. They never see the letters, only the chunks.

Step 3: Train a small model

I trained a tiny GPT-style transformer with 637 184 parameters from a random start. It ran for 60 seconds on my laptop CPU: 811 updates, about 28 000 tokens per second. The corpus has 4 731 tokens but only 1 814 are different, so I used a vocabulary of 1 814 instead of the full 200 019.

Loss started at 7,52, which is what pure guessing gives (ln 1 814 = 7,50). It ended at 0,05. Most of the drop came between update 25 and 150, then it flattened out.

Every sample used the same start ("Don't care about", with a space in front) and the same random seed:

- Before training: random words ("dry Internet Casey ones set feedback neither reason").
- Update 20: only very common tokens ("the the and a at it").
- Update 60 to 140: pieces of different documents mixed together ("foodborne illness", "Health Department", "Brississman-Lundeen Invitational").
- From update 160: it repeats the first document word for word ("Don't care about Sami, really, but hoping that we get some good "SAMANTHA GENE!!" Marlena Death-Stares").

So the model did not learn English in any general way. It memorised the text. It has 637 000 parameters for 4 731 tokens and went through the corpus about 350 times (811 updates x 2 048 tokens / 4 731). The loss is measured on the same text it trained on, so loss alone doesn't prove memorisation. The word for word copies do. Real LLMs train on trillions of tokens, so they can't memorise it all and have to learn patterns instead.

Training also changes the weights. That is the difference from giving a model a report in a chat: the report only goes into the context and the weights stay the same.

Step 3 tests: completing text with the trained model

"The sky is" failed. The tokenizer splits it into "The", " sky" and " is", but " sky" is not in the model's vocabulary, because none of my 10 documents contains it. The model only knows the 1 814 tokens that occur in the corpus, so it can't even read the prompt. A real LLM has seen nearly every token, and it can build rare words from smaller pieces.

Then I let the script pick a sentence from the corpus. It chose "- Put in ONE trash can." from the recycling blog and used "- Put in" as the start. I ran it 100 times with different seeds. 62 of 100 gave the whole original sentence, followed by "By the time you compost, recycle, re-use...". The rest started from the same words but went somewhere else:

- 16 continued with another bullet point from the same list ("a few small bins for various items...").
- A few jumped to a different document ("Augustana, which has only lost to two NCAA Division III schools...").
- One glued pieces together in a way that is not in the text ("Put in you a green can").

"- Put in" is only three tokens and several bullet points start with it, so the model can't know which one I mean. It picks between the places it remembers. This also shows the risk with confidential text in training data: a model can repeat it word for word.

Step 4: Base model vs assistant

I used the SLM Testing space with SmolLM2-135M (base, Completion mode) and SmolLM2-135M-Instruct (Chat mode). Full outputs are in step4_base_vs_instruct.txt. Each prompt ran three times, because the output changes every time.

A base model continues text. "What is 2+2?" turned into a square root question, a blog post and a multiplication lesson. The Paris list turned into invented place names with kilometre numbers, or real landmarks with false history ("Pont Neuf, the first bridge under the Eiffel Tower"). It has no idea it is supposed to answer.

The instruct model does answer, but 135M parameters is small. It said "2 + 2 equals 4" in two of three runs, with nonsense reasoning around it ("2 + 2 = 2 * 2"). One run never gave an answer. One Paris run looped on "the Eiffel Tower, the Arc de Triomphe" until it hit the token limit. It also invented facts, like "Palais Garnier (1994): A former hotel". So instruction tuning changes how the model behaves, not what it knows. I also ran SmolLM2-360M-Instruct by mistake. It was cleaner (4 three times) but still wrote that Versailles is the setting of the film The Godfather.

The Human/Assistant trick: I pasted a short conversation (capital of Norway, days in a leap year) ending in "Human: Why is the sky blue? Assistant:" into the base model. It answered like an assistant, but the answer was wrong (it said blue light cannot be absorbed by our eyes), and then it wrote the next Human and Assistant turns itself. It does not know when its turn ends. The plain "Why is the sky blue?" prompt gave an essay about air bubbles in clouds.

Knowledge cutoff: I used the Literature prize instead of Economics, because it is an easy check. For both 2020 and 2026 the base model said William Faulkner. Faulkner won in 1949 and died in 1962. The 2020 winner was Louise Gluck. For 2026, news sites report Anne Carson, but I could not open nobelprize.org to confirm, so check that yourself. The point is that the model answers confidently about something it can't know, because its data ends earlier.

Other things I noticed: the page says "Loaded ... on CUDA" even though Use GPU was unticked, while the result lines say CPU. I don't know which is true.

Step 5: Supervised fine-tuning data

conversations.json has three questions with their best-ranked answers from OpenAssistant oasst1 (rank 0 among the replies): monopsony in labour markets, contrastive learning, and search engine optimisation. Every message has a message_id so it can be looked up in the dataset.

The answers were written by volunteers and then ranked by other volunteers. Models are fine-tuned to imitate this kind of text. That is also where hallucinations come from. The monopsony answer cites Bivens and Mishel (2013) as research on monopsonies in retail and fast food. I'm fairly sure that paper exists (Journal of Economic Perspectives), but its title is about executive pay and top incomes, so it does not support the claim. A human wrote it, it looked like a good answer, and it was ranked first. A model trained on it learns that answers have a citation at the end, not that the citation has to be right.

Step 6: Hallucinations, thinking and tools

All runs were in claude.ai with Claude Sonnet 5.5, each in a new chat. Full outputs are in step6_hallucinations_thinking_tools.txt.

Obscure article: I picked Benedetto Cotrugli, a merchant from Dubrovnik who described double-entry bookkeeping in 1458, before Pacioli. Without the article the model got 4 of 5 questions right. The wrong one was the coin: it said 25 kuna, the article says 150. It did say it was unsure about that one and told me to check. The library name was also a bit off ("Royal Malta Library" instead of the National Library of Malta). With the article pasted in it got 5 of 5 and corrected itself. So giving the model the source is what fixes it, not asking it nicer.

Who created you: "Anthropic created me. I'm Claude, running as Sonnet 5.5." That matches Anthropic's model documentation, which lists Claude Sonnet 5.5 (claude-sonnet-5-5).

Retailer profit: the right answer is 930,46 euro (I checked it in Python). claude.ai no longer has a thinking on/off switch, only an effort level, so I compared Low effort with High effort on the same model. All four runs said 894,74 euro, so all four were wrong. When it showed the calculation I could see why: it wrote 83% of 1 247 = 1 034,01, but it is 1 035,01. Every step after that was correct. More thinking did not help, because the model still predicts the result of the multiplication as text. Showing the work did help in one way: it made the mistake easy to find. For invoices this is the point. Check totals with code, not with a fluent explanation.

9.11 vs 9.9, strawberry and Ubiquitous: all correct, with and without code. These used to be classic failures, so newer models have probably been trained on them. With code the answer comes from running it, so I can trust it without checking.

Step 7 plan

Run a small instruct model locally in LM Studio and ask for a pelican joke. Screenshot as local_model.png.
