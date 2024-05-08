# Lecture 4 - Feb. 21, 2024

## Housekeeping
- Revisit Pull Requests and Assignment (& my Time Commitment -- will start using CI channel in Discord for notifications)
- [Revisit How to hook in Discord](https://gist.github.com/SGTGunner/50d6a3cc0d489cf779f77695ba3e22ea)
- AzureAI - Unable to assign API keys currently
- GitHub collaborators; Only about half class has added me as collaborator
- Setting up Aider to be delayed until next week (or new commercial sponsor). Still need to find correct IS&T group
- Will be setting up local LLM training w/Aider 
- Class lecture videos will now be posted on permalink
- Quiz is available; Due Monday at midnight
- Where everyone should be:
- * Core repo cloned and forked; ssh keys setup
- * Added GitHub user jeffrey-l-turner as a collaborator to your forked repo
- * Discord CI channel notifications for push pull
- * Pull Request for homework submitted to YOUR fork, not mine...
- * If on Windows, WSL installed and used
- * Should be able to:
```
git pull upstream
git push origin # on your cloned repo to your fork

```
- [Open AI acting up](https://www.independent.co.uk/tech/chatgpt-status-reddit-down-gibberish-messages-latest-b2499816.html) -- Speculation on "Temperature"
```
When one user requested help with a coding issue, the chatbot wrote back a rambling, largely nonsensical answer stating “Let’s keep the line as if AI in the room.” 
```
- Notes from Dr. Vivek Haldar's talk Monday:
- * RAGs and Context Windows
- * Research Papers on Subject
- * Practical elements for students and in industry

## Class Notes - LLMs, Tokenizers, Mixture of Experts, and Code Assistance

Will be experimenting with: [tiktokenizer](https://tiktokenizer.vercel.app/) 

Note Diagram. From [Andrej Karpathy](https://karpathy.ai/) [see](https://www.youtube.com/watch?v=zduSFxRajkE):
```
• Mhy cant LUM spell well? 
• Why cant LLM do super string processing like inverting a string?
• Why are LLMs bad at simple arithmetic? 
• Why dd GPT-2 have more trouble with coding in Python 2? 
• Why did my LLM abruptly halt when it sees "<endoftext>"? 
• What is this weird warnirg about "trailing whitespace"? 
• Why woul I prefer to use YAML over JSON win LLMs? T
• What is the root of my LLM suffering? 
```

## Assignment
- Use [tiktokenizer](https://tiktokenizer.vercel.app/) to with code and place results in (new branch &) pull request (PR). In your own words, describe with the tokenized description how LLMs, tokenizers, mixture of experts, and Coding Assistance fits together. I will be looking for a description in the PR.
- Bonus credit: ask your favorite coding assistant AI to write 3 modules: tokenizer, MoE, and simple neural network to assist with coding. [See Andrej Karpathy's YouTube channel](https://www.youtube.com/watch?v=zduSFxRajkE). Put in a separate PR with different branch.

## References:
- [Transformers, Tokenization, and Embedding](https://vaclavkosar.com/ml/transformer-embeddings-and-tokenization)
- [Type Theory and LLM](https://medium.com/@andrew_johnson_4/harnessing-the-power-of-type-theory-in-large-language-models-351691ca2644) {behind pay wall}
- [Embeddings and LLMs](https://datasciencedojo.com/blog/embeddings-and-llm/)
