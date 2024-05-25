In the context of online continual learning using multi-level online sequential experts, we consider a network composed of $n$ modules, each contributing to the overall learning process. Each module in the network constructs a representation of the input data and aligns these representations to a common dimensional space for consistency across the network.

### Modules and Feature Mapping

- **Feature Mapping**: For each module $i$, we have a feature mapping $d_i$ that transforms the input data into a specific representation.
- **Output Layer**: Denoted by $g_\phi$, this layer generates the final output of the expert.
- **Linear Mapping Layer**: Represented by $p_\psi$, this layer is used in calculating an auxiliary loss for supervision.
- **Dimension Alignment Module**: Denoted by $a_{\omega_i}: \mathbb{R}^{d_i} \mapsto \mathbb{R}^d$, this module projects the feature mapping from the $i$-th module to a consistent $d$-dimensional space. Formally, $\hat{h}_i = a_{\omega_i}(h_i) \in \mathbb{R}^d$.

### Expert Definition

Each expert $E_i$ consists of:

- The corresponding module, 
- The alignment module, and
- The output layer.

### Expert Loss Function

The loss function for each expert $E_i$ combines two types of losses:

1. **Cross-Entropy Loss**: $\mathcal{L}_{ce}(\hat{y}_i, y)$, where $\hat{y}_i$ is the predicted output from the expert and $y$ is the true label.
2. **Supervised Contrastive Loss**: $\mathcal{L}_{scl}(q_i, y)$, which encourages the network to learn discriminative features by leveraging the linear mapping layer output $q_i$.

The loss for each expert $E_i$ is given by:

$$\begin{aligned}
\mathcal{L}_{E_i}(x, y) &= \mathcal{L}_{ce}(\hat{y}_i, y) + \mathcal{L}_{scl}(q_i, y),\\
\text{where, } \hat{y}_i &= E_i(x; \theta_{1:i},\phi) = g_{\phi_i}(a_{\omega_i}(f_{\theta_{1:i}}(x))),\\
q_i &= E_i(x; \theta_{1:i},\psi) = p_{\psi_i}(a_{\omega_i}(f_{\theta_{1:i}}(x))).
\end{aligned}$$

Here, $f_{\theta_{1:i}}(x)$ represents the output of the $i$-th module given the input $x$, and $\theta_{1:i}$ denotes the parameters of modules from 1 to $i$.

### Multi-Level Supervision Loss Function

The overall objective for training involves combining the losses from all experts in the network. This is achieved through the multi-layer supervision loss function $\mathcal{L}_{\text{MLS}}$, which aggregates the individual expert losses over a batch $\mathcal{B}$ of data samples:

$$\mathcal{L}_{\text{MLS}} = \mathbb{E}_{(x_i, y_i) \in \mathcal{B}} \sum_{j=1}^n \mathcal{L}_{E_j}(x_i, y_i).$$

This formulation ensures that each expert contributes to the learning process, and the network benefits from the combined knowledge of all experts. By integrating the outputs and losses across multiple levels, the network is better equipped to handle continual learning tasks.
