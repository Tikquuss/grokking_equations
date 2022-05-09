# OpenAI Grok Curve Experiments

This

```bash
#pip install -r requirements.txt
pip install -e .
# ./scripts/train.py --math_operator s5
math_operator=s5
. train.sh $math_operator
# see train.sh for all the parameters
```

Initially the code supports the version 1.5 of pytorch lightning (see the main branch, setup.py). This branch supports the updated stable version (see setup.py), with the following sensible changes made in the main branch code (grok/training.py):
### Line 609
```python
self.next_train_epoch_to_log = self.next_train_epoch_to_log + 2
```
Instead of :
```python
self.next_train_epoch_to_log = max(
    int(1.01 * self.next_train_epoch_to_log), 
    self.next_train_epoch_to_log + 1
)
```

### Line 705

```python
try : validation_is_real = len(outputs[0]) != 0
except IndexError : validation_is_real = False
```
Instead of :
```python
validation_is_real = len(outputs[0]) != 0
```