**TensorFlow**:

Allows developers to create dataflow graphs: structures that describe how data moves through a graphs

**Benefits**:
Single biggest benefit is *abstraction*
  Instead of dealing with nitty-gritty details of implementing algorithms
  Or figuring out proper ways to hitch the output of one function to the input of another
  TensorFlow takes care of these details behind the scenes, developer can focus on overall logic of application

Eager execution mode allows one to evaluate and modify each graph operation separately and transparently
  Instead of constructing the entire graph as a single opaque object and evaluating it all at once

TensorBoard visualization suite lets you inspect and profile the way graphs run by way of an interactive web-based dashboard

If use google's own cloud, can run TensorFlow on google's custom Tensor Processing Unit (TPU) for further acceleration

**Drawback**:
Sometimes a model trained on one system will vary slightly from a model trained on another, even when fed the same dataflow
    Could be due to how random numbers are seeded and where
    Possible to work around these issues

**Competitors**:

PyTorch: Similar to TensorFlow, generally better choice for fast development of projects that need to be up and running in a short time, TensorFlow wins out for larger projects and more complex workflows

CNTK: Like TensorFlow uses a graph structure to describe dataflow
  Focuses more on creating deep learning neural networks
  Not currently as easy to learn or deploy as TensorFlow

Apache MXNet: Can scale almost linearly across multiple GPUs and multiple machines
  Native APIs not as pleasant as TensorFlow's

**PyTorch**:

Praised for it's "pythonic" nature by python enthusiasts
  Thus fits smoothly into the Python ML ecosystem

TensorFlow has interfaces in many programming languages
  The High-level Keres API for TensorFlow in Python has proven so successful that newest TensorFlow version integrates it by default
  Keres interface offers readymade building blocks, significantly improves speed

TensorFlow: graph is defined statically
  Outline entire structure its entire structure, layers and connections, what kind of data gets processed, before running it
  Graph cannot be modified after compilation
  TensorFlow 2.0 now has eager execution by default, too

PyTorch: Defined dynamically
  Graph and its input can be modified during runtime -> eager execution
    Offers programmer better access to inner workings of the network than a static graph
    Eases process of debugging code

TensorFlow has visualization feature: TensorBoard
  To view neural networks as a graph with nodes and edges
  Allows one to observe behavior of training parameters over time, by logging summaries at predefined intervals
  Valuable for debugging

PyTorch does not have native visualization feature
  Uses regular python packages like matplotlib or seaborn for plotting
  Visualization packages available, do not display same versatility as TensorBoard

TensorFlow makes it easy to offer and update your trained models on server-side, possible for PyTorch

PyTorch long been preferred deep-learning library for researchers, TensorFlow is much more widely used in production
  PyTorch ease of use combined with default eager execution mode for easier debugging predestines it to be used for fast, smaller-scale models
  TensorFlow extensions for deployment on both servers and mobile devices, makes it preferred for company use

**Conclusion**

PyTorch:
1. "Pythonic" package, easy to use, fits nicely into python ecosystem
2. Default eager execution-mode leads to easier debugging
3. No visualization feature, uses matplotlib or seaborn or etc...

TensorFlow:
1. Easy to offer and update trained models server-side, interfaces in many languages, better if desire to provide ML for general use
2. Visualization package "TensorBoard", view neural networks as graph with nodes and edges, view training parameters over time
3. If use Google's own cloud, can run TensorFlow on Google's custom Tensor Processing Unit (TPU) for further acceleration
