Notes to include in the README.md at some point:
* first failed experiment: gpt-2-large as the target model and gpt-2(small) as the draft model. They ended up looping at any non trivial prompt.
* Tried using hand written prompts, wiki headlines and even hand-written prompts.
* Switched to the Pythia family by EleutherAI. The target model will be the 1.4b parameters version and the draft model will be the 160M parameters version.
* Models from the same training family are the canonical setup for speculative decoding experiments, as they maximize draft-target alignment and isolate the algorithm's behavior from distribution mismatch noise.