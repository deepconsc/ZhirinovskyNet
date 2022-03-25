# ZhirinovskyNet
ResNet-50 Encoder-based modelling for Schrödinger's Zhirinovsky Paradox

!["Zhirinovsky and some unknown man."](https://c8.alamy.com/zooms/9/47f382ffccf04c1ebf669a14354d4957/tyfbac.jpg "Zhirinovsky and some unknown man.") 
******
***Image: Zhirinovsky and some unknown man.***

In the name of recent events, we aim to compute the probability of whether Zhirinovsky-  otherwise known as Squealer is alive or dead. 

## Approach
We use ImageNet-pretrained ResNet-50 encoder to compute probs over [341th class](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a#file-imagenet1000_clsidx_to_labels-txt-L342) using randomly sampled images of Zhirinovsky.

As long as Russian media propaganda tries to change the outcome every day, we add up some normal noise to the input to simulate the superposition of Zhirinovsky. 

As an author, I wish the real-world probability is less than 0.45.

**Glory to Ukraine!** 🇺🇦


## Instructions

Install the requirements:

```
python3 -m pip install -r requirements.txt
```

Run the evaluation with 1 trial:

```
python3 eval.py 
```

Or run the evaluation with N trials, utilizing gpu.


```
python3 eval.py --trials N --device cuda
```

Example output:
```
5 Trial Runs have been finished.
Congratulations! Theoretically he turns out to be DEAD with the probability of 0.83.
```