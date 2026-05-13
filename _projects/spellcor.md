---
name: SpellCor
description: A fully Python spell correction tool inspired by JamSpell. Supports custom language models, dictionary filtering, and n-gram language model training. Pip-installable with extensible architecture.
tags: [NLP, Spell Correction, Python]
url: https://github.com/xvshiting/SpellCor
github: xvshiting/SpellCor
highlight: true
---

## Overview

SpellCor is a spell correction tool coded fully with Python, inspired by [JamSpell](https://github.com/bakwc/JamSpell) (written in C++ with Python bindings via swig). While JamSpell is efficient due to its C++ implementation, it's not convenient for developers who want to extend or customize features. Considering the prevalence of Python in the NLP community, SpellCor provides a pure Python alternative with greater flexibility.

## Why SpellCor?

- **Fully Python** — Includes a Python n-gram language model, no C++ dependency
- **Extensible** — Easily plug in your own language model (e.g., NN models) with a simple interface
- **Dictionary filtering** — Filter candidate words using your own clean dictionary
- **Pip-installable** — `pip install spellcor`

## Installation

```bash
pip install spellcor
```

Train your own n-gram language model:

```bash
python -m spellcor data.txt output_model_dir
```

## Basic Usage

```python
import spellcor

checker = spellcor.SpellChecker()

# Load language model
model_name = "BaseLanguageModel"
model_path = "./tmp/nglm.bin"
checker.load_lang_model(model_name, model_path)

# Sentence correction
checker.fix_sentence("here are some Questino , I am pyspell checkre")
# → "here are some Question , I am spell checker"

# Fix by position
checker.fix_pos(["here", "are", "some", "Questino"], 3)
# → ['Question', 'Questino']
```

## Extending Language Models

Create your own model by subclassing `AbstractLanguageModel`:

```python
import spellcor

@spellcor.register_lang_model("NewModel")
class MyModel(spellcor.AbstractLanguageModel):
    def __init__(self, model_path, **kwargs):
        super(MyModel, self).__init__("NewModel")
        self.language_model = None
        self.load_lang_model(model_path)

    def get_score(self, sentence, word_start_index, word_end_index):
        pass

    def is_word(self, word):
        return self.language_model.is_word(word)

    def load_lang_model(self, model_path):
        pass

    def word_freq(self, word):
        pass
```

## Dictionary Filtering

```python
# Load a clean dictionary to filter candidates
checker.load_valid_word_dict(dict_path)

# Or filter candidate words separately
checker.load_valid_candidate_dict(dict_path)

# Toggle dictionary usage
checker.use_valid_word_dict = False
checker.use_valid_candidate_dict = True
```

## Citation

If you use SpellCor in your research, please cite:

```bibtex
@software{spellcor,
  author = {Xu, Shiting},
  title = {SpellCor: A Fully Python Spell Correction Tool},
  url = {https://github.com/xvshiting/SpellCor}
}
```