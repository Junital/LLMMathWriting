The intermediate GSS in NSR is both latent and non-differentiable, making direct application of backpropagation unfeasible. Traditional approaches, such as policy gradient algorithms like REINFORCE (Williams, 1992), face difficulties with slow or inconsistent convergence (Liang et al., 2018; Li et al., 2020). Given the vast search space for GSS, a more efficient learning algorithm is imperative. Formally, for input $x$, intermediate GSS $T=<(x, s, v), e>$, and output $y$, the likelihood of observing $(x, y)$, marginalized over, $T$, is expressed as:

$$ p(y|x;\Theta) = \sum_T p(T, y|x;\Theta) = \sum_{s, e, v} p(s|x; \theta_p) p(s|x; \theta_s) p(s|x; \theta_l) p(y|v), $$

where maximizing the observed-data log-likelihood $L(x, y) = \log p(y|x)$ becomes the learning objective, from a maximum likelihood estimation viewpoint. The gradients of $L$ with respect to $\theta_p, \theta_s, \theta_l$ are as follows:

$$ \begin{aligned}
\nabla_{\theta_p} L(x, y) &= \mathbb{E}_{T \sim p(T|x, y)} \left[ \nabla_{\theta_p} \log p(s|x; \theta_p) \right], \\
\nabla_{\theta_s} L(x, y) &= \mathbb{E}_{T \sim p(T|x, y)} \left[ \nabla_{\theta_s} \log p(s|x; \theta_s) \right], \\
\nabla_{\theta_l} L(x, y) &= \mathbb{E}_{T \sim p(T|x, y)} \left[ \nabla_{\theta_l} \log p(s|x; \theta_l) \right].
\end{aligned} $$

where $p(T|x, y)$ denotes the posterior distribution of $T$ given $(x, y)$, which can be represented as:

$$ p(T|x,y) = \frac{p(T, y|x;\Theta)}{\sum_{T'} p(T', y|x;\Theta)} = \begin{cases} 
0, & \text{if } T \notin Q, \\
\frac{p(T|x; \Theta)}{\sum_{T' \in Q} p(T'|x; \Theta)}, & \text{if } T \in Q,
\end{cases} $$

with $Q$ as the set of $T$ congruent with $y$.
