## CPPNs and differentiable implementation

Summary and references:
**1) NEAT (NeuroEvolution of Augmenting Topologies):** while previous work proposed genetic algorithms to evolve the weights of fixed-topologt NN, NEAT is a neuroevolution algorithm that evolves the architectures of its networks in addition to the weights. See [Stanley et al., 2002 paper](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
**2) CPPN (Compositional Pattern Producing Network):**  CPPN is a recurrent NN (generally small with few neurons/connections) of the form *f(input,structural bias)=output* which is designed to represent patterns with regularities such as symmetry, repetition, and repetition with variation. CPPN must fully activate once for each coordinate in the phenotype, making its complexity O(N) (O(N2) for images). See [Stanley 2007 paper](https://eplex.cs.ucf.edu/papers/stanley_gpem07.pdf).
**3) CPPN-NEAT:**  Slight modifications of NEAT (several activation functions instead of one) to evolve CPPN networks. CPPN are small encoding networks (resolution independant) making the NEAT optimization less complex than its previous application on big network phenotypes.  

**Applications of CPPN-NEAT**: while it was initially proposed to evolve 2D image phenotypes, the output of the CPPN network can take many forms and other applications have been proposed:
* **HyperNEAT** propose to use CPPN to indirectly encode the weights of the network (called the *substrate*) and to evolve it with CPPN-NEAT. See [Stanley et al. 2009 paper](https://www.researchgate.net/publication/23986881_A_Hypercube-Based_Encoding_for_Evolving_Large-Scale_Neural_Networks) and [website](http://eplex.cs.ucf.edu/hyperNEATpage/). ![image|680x337, 75%](upload://1JTJ9HYXdplitaLBLVCYBXKcu4X.png) 
* **Evolving soft-robot morphologies**: while the output was previously considered as continuous and 1-dimensional(image intensity for CPPN or connection weight in HyperNEAT), the output can be modularized and divided into binary/continuous outputs as proposed in [Cheney et al. 2014](https://www.researchgate.net/publication/270696982_Unshackling_evolution) ![image|475x303, 75%](upload://oSozteSpD3iHNUie7dwPs0Zndiy.png) 
* Other applications have considered interactive evolution-schemes implicating humans to select  phenotypes and use this as fitness function to evolve "interesting" 2D or 3D patterns (see [picbreeder](http://picbreeder.org/) and [endlessforms](http://endlessforms.com/)

**Implementations**:
Official repositories are [NEAT-Python](https://neat-python.readthedocs.io/en/latest/) and [PyTorch-Neat](https://github.com/uber-research/PyTorch-NEAT) by Uber Research. 
While the second repo is based on torch tensors allowing to use GPU-acceleration, it does not implement the phenotype networks as torch.nn.module and do not provide example use cases of differentiating though the network evolved weights. 

**Differentiable CPPNs**:
While the idea of evolving the network topologies with NEAT algo and the network weights with gradient descent has been proposed in previous papers (see [DPPN paper](https://arxiv.org/pdf/1606.02580.pdf) for instance), I could not find any code doing it. Here is a code which:
1) Reproduce the results of the official NEAT-Python package: see tests/test_differentiable_cppn.py
2) Implements the sigmoid, tanh, gauss, abs, relu, sin and identity activations as well as the sum and product aggregation functions
3) Is differentiable: see tests/test_rnn.py
