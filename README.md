# Min-Llama 

1) Generates a text completion (starting with the sentence `"I have wanted to see this thriller for a while, and it didn't disappoint. Keanu Reeves, playing the hero John Wick, is"`). You should see coherent, grammatical English being generated (though the content and topicality of the completion may be absurd, since this LM was pretrained exclusively on children's stories).
2) Performed zero-shot, prompt-based sentiment analysis on two datasets (SST-5 and CFIMDB). This will give bad results (roughly equal to choosing a random target class).
3) Performed task-specific finetuning of your Llama2 model, after implementing a classification head in `classifier.py`. This will give much stronger classification results.

## Important Notes
* Follow `setup.sh` to properly setup the environment and install dependencies.
* There is a detailed description of the code structure in [structure.md](./structure.md), including a description of which parts you will need to implement.
* You are only allowed to use libraries that are installed by `setup.sh`, no other external libraries are allowed (e.g., `transformers`).
* We will run your code with commands below (under "Reference outputs/accuracies"), so make sure that whatever your best results are reproducible using these commands.
    * Do not change any of the existing command options (including defaults) or add any new required parameters

## Reference outputs/accuracies: 

*Text Continuation* (`python run_llama.py --option generate`)
You should see continuations of the sentence `I have wanted to see this thriller for a while, and it didn't disappoint. Keanu Reeves, playing the hero John Wick, is...`. We will generate two continuations - one with temperature 0.0 (which should have a reasonably coherent, if unusual, completion) and one with temperature 1.0 (which is likely to be logically inconsistent and may contain some coherence or grammar errors).

*Zero Shot Prompting*
Zero-Shot Prompting for SST:

`python run_llama.py --option prompt --batch_size 10  --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt --label-names data/sst-label-mapping.json --dev_out sst-dev-prompting-output.txt --test_out sst-test-prompting-output.txt [--use_gpu]`

Prompting for SST:
Dev Accuracy: 0.213 (0.000)
Test Accuracy: 0.224 (0.000)

Zero-Shot Prompting for CFIMDB:

`python run_llama.py --option prompt --batch_size 10  --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt --label-names data/cfimdb-label-mapping.json --dev_out cfimdb-dev-prompting-output.txt --test_out cfimdb-test-prompting-output.txt [--use_gpu]`

Prompting for CFIMDB:
Dev Accuracy: 0.498 (0.000)
Test Accuracy: -

*Classification Finetuning*

`python run_llama.py --option finetune --epochs 5 --lr 2e-5 --batch_size 80  --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt --label-names data/sst-label-mapping.json --dev_out sst-dev-finetuning-output.txt --test_out sst-test-finetuning-output.txt [--use_gpu]`

Finetuning for SST:
Dev Accuracy: 0.414 (0.014)
Test Accuracy: 0.418 (0.017)

`python run_llama.py --option finetune --epochs 5 --lr 2e-5 --batch_size 10  --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt --label-names data/cfimdb-label-mapping.json --dev_out cfimdb-dev-finetuning-output.txt --test_out cfimdb-test-finetuning-output.txt [--use_gpu]`

Finetuning for CFIMDB:
Dev Accuracy: 0.800 (0.115)
Test Accuracy: -

Mean reference accuracies over 10 random seeds with their standard deviation shown in brackets.

