#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import ipywidgets as widgets
import torch
from nltk.corpus import stopwords
from transformers import (
    AutoModel,
    AutoTokenizer
)
from IPython.core.display import display, HTML


# In[2]:


from .utils import find_positions, make_the_words, get_sample_article, scale, make_html


# In[3]:


class AttentionVisualizer():
    
    def __init__(self):
        super().__init__()
        
        self.model_name = "roberta-base"
        self.model      = AutoModel.from_pretrained(self.model_name)
        self.tokenizer  = AutoTokenizer.from_pretrained(self.model_name)
        self.stop_words   = list(stopwords.words('english'))

        ###############
        # UI Elements #
        ###############
        self.step1_lbl = widgets.HTML(value = f"<h3>Step 1:</h3> Write the input text in the box below.")
        self.input_text = widgets.Textarea(placeholder='Type something', description='Input Text:', rows=10)
        self.step1 = widgets.VBox([self.step1_lbl, self.input_text])
        
        self.preview_config_lbl = widgets.HTML(value = f"<h3>Step 2:</h3> Select which attention layer/head and words to visualize.")
        self.ignore_specials  = widgets.Checkbox(value=True, description='Ignore BOS/EOS', indent=False)
        self.ignore_dots     = widgets.Checkbox(value=True, description='Ignore [dot]s', indent=False)
        self.ignore_stopwords = widgets.Checkbox(value=True, description='Ignore Stop Words', indent=False)
        self.options = widgets.HBox([self.ignore_specials, self.ignore_dots, self.ignore_stopwords])
        
        self.layer_range = widgets.IntRangeSlider(value=[3, 5], min=1, max=12, step=1,
                                continuous_update=False, orientation='horizontal', readout=True,
                                readout_format='d')
        self.layer_ind = widgets.Dropdown(options=range(1, 13), value=1)
        self.layer = widgets.Dropdown(options=['all', 'range', 'individual'], value='all', description='Layer')
        self.layer_selection = widgets.HBox([self.layer, self.layer_range, self.layer_ind])

        self.head_range = widgets.IntRangeSlider(value=[3, 5], min=1, max=12, step=1,
                                continuous_update=False, orientation='horizontal', readout=True,
                                readout_format='d')
        self.head_ind = widgets.Dropdown(options=range(1, 13), value=1)
        self.head = widgets.Dropdown(options=['all', 'range', 'individual'], value='all', description='Head')
        self.head_selection = widgets.HBox([self.head, self.head_range, self.head_ind])
        
        self.visualize_btn = widgets.Button(description='VISUALIZE')
        self.note_lbl = widgets.HTML(value = f"<small>Hold your cursor on each word for a second to see its attention score.</small>")

        self.out = widgets.HTML(layout={'border': '1px solid black', 'padding': '4px', 'margin-top': '10px'})
        
        self.step2 = widgets.VBox([self.preview_config_lbl, self.options,
                                   self.layer_selection, self.head_selection, self.visualize_btn,
                                   self.note_lbl, self.out])
        
        self.ui = widgets.VBox([self.step1, self.step2])

    def show_controllers(self, with_sample=False):
        # Hide the options
        self.hide(self.layer_range)
        self.hide(self.layer_ind)
        self.hide(self.head_range)
        self.hide(self.head_ind)
        
        # Register on_click Drop Down
        self.layer.observe(self.on_dd_change)
        self.head.observe(self.on_dd_change)
        
        # Register on_click Visualize button
        self.visualize_btn.on_click(self.on_visualize_click)
        
        if with_sample:
            self.input_text.value = get_sample_article()
        
        return display(self.ui)
    
    def on_visualize_click(self, c):
        self.out.value =""
        
        inputs = self.tokenizer(self.input_text.value, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model(**inputs, output_attentions=True)
        
        the_tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0],
                                                          skip_special_tokens=self.ignore_specials.value)
        number_of_tokens = inputs.input_ids.size(1)
        positions, dot_positions, stopwords_positions = find_positions(self.ignore_specials.value,
                                                                       self.ignore_stopwords.value,
                                                                       the_tokens,
                                                                       self.stop_words)
        the_words = make_the_words(self.input_text.value, positions, self.ignore_specials.value)
        
        layer_indexes, head_indexes = self.extract_indexes()
        
        the_scores = []
        for i in range(*layer_indexes):
            for ii in range(*head_indexes):
                the_scores.append( torch.sum(outputs.attentions[i][0][ii], dim=0) / number_of_tokens )
        
        the_scores = torch.stack( the_scores )
        final_score = torch.sum( the_scores, dim=0 ) / the_scores.size(0)
        
        # Remove the CLS/SEP tokens and dots
        if self.ignore_specials.value:
            final_score = final_score[1:-1]
            
        min_ = torch.min( final_score )
        
        if self.ignore_dots.value:
            final_score[list(dot_positions.values())] = min_

        if self.ignore_stopwords.value:
            final_score[list(stopwords_positions.values())] = min_
        
        max_ = torch.max( final_score )
        
        for i in range( final_score.size(0) ):
            final_score[i] = scale( final_score[i], min_, max_ )
        
        the_html = make_html(the_words, positions, final_score)
        self.out.value = the_html
        
        
    def on_dd_change(self, change):
        
        if change['type'] == 'change' and change['name'] == 'value':
            if change.owner.description == "Layer":
                ind_dd = self.layer_ind
                slider = self.layer_range
            elif change.owner.description == "Head":
                ind_dd = self.head_ind
                slider = self.head_range
            else:
                raise NotImplementedError
                
            if change['new'] == "individual":
                ind_dd = self.show(ind_dd)
                slider = self.hide(slider)

            elif change['new'] == 'range':
                ind_dd = self.hide(ind_dd)
                slider = self.show(slider)

            else:
                ind_dd = self.hide(ind_dd)
                slider = self.hide(slider)
    
    def extract_indexes(self):
        layer = self.layer.value
        head  = self.head.value

        # Get The Layers Indexes
        if layer == "all":
            layer_indexes = 0, 12
        elif layer == "range":
            layer_indexes = self.layer_range.value[0]-1, self.layer_range.value[1]
        elif layer == "individual":
            layer_indexes = self.layer_ind.value-1, self.layer_ind.value

        # Get The Heads Indexes
        if head == "all":
            head_indexes = 0, 12
        elif head == "range":
            head_indexes = self.head_range.value[0]-1, self.head_range.value[1]
        elif head == "individual":
            head_indexes = self.head_ind.value-1, self.head_ind.value
        
        return layer_indexes, head_indexes
        
    def hide(self, el):
        el.layout.visibility = "hidden"
        return el

    def show(self, el):
        el.layout.visibility = None
        return el
