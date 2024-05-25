Based on the above dimension alignment of multi-scale features, it becomes possible to train each block of the network as a fully functional continual learner. To do so, we add output heads after each block to transform the projected feature $\hat{h}_i$ into an output vector for supervised loss computation. The choice of output head depends on the supervision loss we use, and this multi-level framework is generally compatible with most replay-based methods. Shown in Fig. 2, we add two types of output heads after all alignment module $a_{\omega_i}: p_{\psi_i}$ and $g_{\phi_i}$ for supervised contrastive representation learning and cross-entropy-based classification learning (see Sec. 3.2), respectively. We then mark them as latent experts $E_i$, consist of sequential blocks $\{f_{\theta_j}\}_{j\le i}$ up to block $i$ and corresponding feature alignment module $a_{\omega_i}$ and output heads $p_{\psi_i}$ and $g_{\phi_i}$. Naturally, each expert is trained with compounded supervision loss $\mathcal{L}_{E_i}$:

$$\begin{aligned}
\mathcal{L}_{E_i}(x, y) &= \mathcal{L}_{ce}(\hat{y}_i, y) + \mathcal{L}_{scl}(q_i, y),\\
\text{where, } \hat{y}_i &= E_i(x; \theta_{1:i},\phi) = g_{\phi_i}(a_{\omega_i}(f_{\theta_{1:i}}(x))),\\
q_i &= E_i(x; \theta_{1:i},\psi) = p_{\psi_i}(a_{\omega_i}(f_{\theta_{1:i}}(x))).
\end{aligned}$$

Therefore, Multi-Level Supervision (MLS) signal $\mathcal{L}_{\text{MLS}}$ is injected into each block of the network by summing up expert-wise losses and this framework fits perfectly with most replay-based methods:

$$\mathcal{L}_{\text{MLS}} = \mathbb{E}_{(x_i, y_i) \in \mathcal{B}} \sum_{j=1}^n \mathcal{L}_{E_j}(x_i, y_i).$$
