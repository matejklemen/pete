```python
from explain_nlp.experimental.core import MethodData, MethodGroup

# STEP 1
# Option 1a) don't load methods manually
# ...

# Option 1b) load methods manually
method1 = MethodData.load("path1.json")
method2 = MethodData.load("path2.json")
method3 = MethodData.load("path3.json")

# STEP 2
# Option 2a) load data for different methods into a common data group
experiment = MethodGroup(["path1.json", "path2.json", "path3.json"], method_labels=["method_name1", "method_name2", "baseline"])

# Option 2b) manually set methods and their aliases in plots
experiment = MethodGroup([], [])
experiment.methods = [method1, method2, method3]
experiment.method_labels = ["method_name1", "method_name2", "baseline"]

# STEP 3
experiment.plot_required_samples()              # scatterplot
experiment.plot_required_samples_wins()         # heatmap/confusion matrix
experiment.plot_required_samples_avg_diff()     # heatmap/confusion matrix
experiment.plot_example_mae()                   # scatterplot
experiment.plot_global_mae()                    # barplot
```