**Neural-Symbolic Recursive Machine (NSR) for Systematic Generalization**

Current learning models often struggle with human-like systematic generalization, particularly in learning compositional rules from limited data and extrapolating them to novel combinations. To address this challenge, we introduce the Neural-Symbolic Recursive Machine (NSR), a model designed to facilitate systematic generalization through a Grounded Symbol System (GSS). The GSS enables the emergence of combinatorial syntax and semantics directly from training data, thus allowing the NSR to efficiently handle a variety of sequence-to-sequence tasks.

### Grounded Symbol System (GSS)

At the core of NSR is the GSS, which can be represented as a directed tree $T=<(x, s, v), e>$, where:

- $x$ is the grounded input,
- $s$ is an abstract symbol,
- $v$ is its semantic meaning,
- $e$ denotes semantic dependencies, with directed edges $i \rightarrow j$ indicating that the meaning of node $i$ depends on node $j$.

### Likelihood and Learning Objective

The likelihood of observing an output $y$ given an input $x$ is expressed as:

$$ p(y|x;\Theta) = \sum_T p(T, y|x;\Theta) = \sum_{s, e, v} p(s|x; \theta_p) p(s|x; \theta_s) p(s|x; \theta_l) p(y|v), $$

where $\Theta = \{\theta_p, \theta_s, \theta_l\}$ represents the model parameters across different components:

- $\theta_p$ for perception,
- $\theta_s$ for syntactic parsing,
- $\theta_l$ for semantic reasoning.

The learning objective in the context of maximum likelihood estimation is:

$$ L(x, y) = \log p(y|x). $$

### Gradient Calculation

The gradients of $L(x, y)$ with respect to the parameters $\theta_p$, $\theta_s$, and $\theta_l$ are computed as follows:

$$ \begin{aligned}
\nabla_{\theta_p} L(x, y) &= \mathbb{E}_{T \sim p(T|x, y)} \left[ \nabla_{\theta_p} \log p(s|x; \theta_p) \right], \\
\nabla_{\theta_s} L(x, y) &= \mathbb{E}_{T \sim p(T|x, y)} \left[ \nabla_{\theta_s} \log p(s|x; \theta_s) \right], \\
\nabla_{\theta_l} L(x, y) &= \mathbb{E}_{T \sim p(T|x, y)} \left[ \nabla_{\theta_l} \log p(s|x; \theta_l) \right].
\end{aligned} $$

Here, $p(T|x, y)$ is the posterior distribution of tree $T$ given $(x, y)$.

### Posterior Distribution of the Tree

The posterior distribution $p(T|x, y)$ is defined as:

$$ p(T|x,y) = \frac{p(T, y|x;\Theta)}{\sum_{T'} p(T', y|x;\Theta)} = \begin{cases} 
0, & \text{if } T \notin Q, \\
\frac{p(T|x; \Theta)}{\sum_{T' \in Q} p(T'|x; \Theta)}, & \text{if } T \in Q,
\end{cases} $$

where $Q$ is the set of trees $T$ that are congruent with $y$.

### Integration and Training

The NSR employs a modular design that integrates neural perception, syntactic parsing, and semantic reasoning, trained synergistically through a novel deduction-abduction algorithm. By imbuing the model with inductive biases of equivariance and compositionality, the NSR achieves unparalleled systematic generalization, addressing the limitations of current learning models in handling novel combinations and compositional rules.
