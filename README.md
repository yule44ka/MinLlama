# Min-Llama 

1) Generates a text completion (starting with the sentence `"I have wanted to see this thriller for a while, and it didn't disappoint. Keanu Reeves, playing the hero John Wick, is"`). You should see coherent, grammatical English being generated (though the content and topicality of the completion may be absurd, since this LM was pretrained exclusively on children's stories).
2) Performed zero-shot, prompt-based sentiment analysis on two datasets (SST-5 and CFIMDB). This gives bad results (roughly equal to choosing a random target class).
3) Performed task-specific finetuning of your Llama2 model and implemented head in `classifier.py`. This gives much stronger classification results.

## Important Notes
* Follow `setup.sh` to properly setup the environment and install dependencies.
* There is a detailed description of the code structure in [structure.md](./structure.md).
