
# Attention Visualizer

<p align="center">
<a href="https://raw.githubusercontent.com/AlaFalaki/AttentionVisualizer/main/images/output.png">
<img width="250" src="https://raw.githubusercontent.com/AlaFalaki/AttentionVisualizer/main/images/output.png" />
</a>
<a href="https://raw.githubusercontent.com/AlaFalaki/AttentionVisualizer/main/images/UI.png"">
<img width="250" src="https://raw.githubusercontent.com/AlaFalaki/AttentionVisualizer/main/images/UI.png">
</a>
</p>

A fun project that turns out as a python package with a simple UI to visualize the self-attention score using the RoBERTa library. It is implemented for IPython Notebook environment with options to ignore tokens like "BOS", "[dot]s", or "stopwords". You can also look at a range or specific Layers/Heads.
 
## Demo

Run the library on a Google Colab instance using the [following link](https://colab.research.google.com/github/AlaFalaki/AttentionVisualizer/blob/main/demo.ipynb).

:warning: *There is a bug (only happens in Google Colab) that the instance's memory usage will increase everytime a new request is sent. Would be happy to know what the problem could be if anyone encounter it before.*

## Installation

The package is only hosted on Github for now. You can still use `pip` to install the package. 

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


## Implementation Details

If you are interested in the project and want to know more, I wrote a blog post on medium that explain the implementation in detail. 

[Attention Visualizer Package: Showcase Highest Scored Words Using RoBERTa Model](https://pub.towardsai.net/attention-visualizer-package-showcase-highest-scored-words-using-roberta-model-8218658b4447)

