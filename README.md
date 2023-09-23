
<a href="https://twitter.com/NLPiation">
<img src="https://img.shields.io/badge/Author-Twitter-blue" />
</a>
<a href="https://pub.towardsai.net/attention-visualizer-package-showcase-highest-scored-words-using-roberta-model-8218658b4447">
<img src="https://img.shields.io/badge/Blog%20Post-Medium-orange" />
</a>
<a href="https://colab.research.google.com/github/AlaFalaki/AttentionVisualizer/blob/main/demo.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" />
</a>

# Attention Visualizer

<p align="center">
<img width="350" src="https://raw.githubusercontent.com/AlaFalaki/AttentionVisualizer/main/images/output.png" />
<img width="350" src="https://raw.githubusercontent.com/AlaFalaki/AttentionVisualizer/main/images/UI.png">
</p>

A fun project that turns out as a python package with a simple UI to visualize the self-attention score using the RoBERTa library. It is implemented for IPython Notebook environment with options to ignore tokens like "BOS", "[dot]s", or "stopwords". You can also look at a range or specific Layers/Heads.
 
## Demo

Run the library on a Google Colab instance using the [following link](https://colab.research.google.com/github/AlaFalaki/AttentionVisualizer/blob/main/demo.ipynb).

:warning: *There is a bug (only happens in Google Colab) that the instance's memory usage will increase everytime a new request is sent. Would be happy to know what the problem could be if anyone encounter it before.*

## Installation

The package is only hosted on Github for now. You can use `pip` to install the package. 

```bash
  pip install git+https://github.com/AlaFalaki/AttentionVisualizer.git
```


## Usage/Examples

Run the code below in an IPython Notebook.

```python
import AttentionVisualizer as av

obj = av.AttentionVisualizer()
obj.show_controllers(with_sample=True)
```


## Requirements

The package will automatically installs all the requirements.

- pytorch
- transformers
- ipywidgets
- NLTK


## Implementation Details

If you are interested in the project and want to know more, I wrote a blog post on medium that explain the implementation in detail. 

<a href="https://pub.towardsai.net/attention-visualizer-package-showcase-highest-scored-words-using-roberta-model-8218658b4447">
Attention Visualizer Package: Showcase Highest Scored Words Using RoBERTa Model
</a>

## Citation

arXiv preprint: <a href="https://arxiv.org/abs/2308.14850">Attention Visualizer Package: Revealing Word Importance for Deeper Insight into Encoder-Only Transformer Models</a>

```
@article{falaki2023attention,
  title={Attention Visualizer Package: Revealing Word Importance for Deeper Insight into Encoder-Only Transformer Models},
  author={Falaki, Ala Alam and Gras, Robin},
  journal={arXiv preprint arXiv:2308.14850},
  year={2023}
}
```
