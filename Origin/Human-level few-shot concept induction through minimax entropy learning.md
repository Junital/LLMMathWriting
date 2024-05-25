Instead of formulating concept induction as a discriminative process, minimax entropy learning frames the problem as a descriptive process (22). Specifically, for each problem’s context set $C = \{x_i\}$, consisting of object-centric representation of either two to three short sequences or static images, our ME model should learn a distribution $p(x)$ on the object-centric representation space that best characterizes the hidden concept while making unconstrained dimensions as random as possible. Formally, assuming the hidden concept can be captured by a set of response functions $\{H_j(\cdot)\}$ or filters, the intuition aforementioned can be formulated as the following maximum entropy principle

$$\begin{aligned}
\max_p &-\int p(x) \log p(x) \, \mathrm{d}x \\
\text{subject to} &\underset{x \sim p(x)}{\mathbb{E}}[H_j(x)] = \mu_j^{\text{obs}}, \quad \forall j \\
&\int p(x) \, \mathrm{d}x = 1,
\end{aligned}$$

where $\mu_j^\text{obs}$ denotes the average filter response on the context panels. This formulation requires the dimensions captured by the filters to match the observed statistics while setting others as unrestrained as possible. The optimization bears an analytical solution

$$
p(x) = \frac{1}{Z} \exp\left[-\sum_j \lambda_j H_j(x)\right],
$$

where $Z = \int \exp\left[-\sum_j \lambda_j H_j(x)\right] \, \mathrm{d}x$ is the normalizer and λj are the optimal Lagrangian multipliers that can be learned through maximum likelihood learning on Eq. 2 (23). In the minimum entropy stage, we minimize the entropy of the model on the basis of the maximum entropy results, which is equivalent to minimizing the KL divergence between the true distribution constrained by the hidden concepts $p^\star(x)$ and our approximation $p(x)$

$$
\underset{p}{\min} -\int p(x) \log p(x) \, \mathrm{d}x = \underset{p}{\min} \mathbf{KL}(p^*, p) = \underset{p}{\max} \mathbb{E}_{p^*}[\log p(x)].
$$

Note that the formulation is also equivalent to maximum likelihood as shown above. This step is implemented as selecting the optimal set of filters $\{H_j(\cdot)\}$ among others to minimize the expected coding length under the coding scheme of the chosen filters (23). Unlike the greedy method of feature pursuit (23), we explicitly add a set of global indicator variables $\{z_j\}$ to the optimization and alternatively maximize the log likelihood of the distribution, that is

$$
\max_{\lambda, z} \mathbb{E}_{x_i}[\log p(x_i)] = \mathbb{E}_{x_i}\left[-\sum_j \lambda_j z_j H_j(x_i) - \log Z\right],
$$

There is still a remaining issue with the minimax entropy learning framework aforementioned: The traditional fixed filter design is limited in expressiveness and cannot adapt to different cases in distinctive scenarios. Adding more filters can potentially mitigate this issue, but a large number of filters will unnecessarily complicate the minimax learning process. Besides, for relational concepts in continuous spaces, capturing a unique one with a finite number of fixed filters is difficult. To address this issue, we further devise to parameterize the filter functions and solve for the optimal one based on the context onthe-fly. Specifically, we embed the minimax entropy learning framework inside a bilevel optimization problem (28–30): The inner-level optimization works out the optimal parameters corresponding to the best filters in continuous filter families to felicitously describe the hidden concepts, and the outer-level optimization performs the minimax entropy learning steps aforementioned. Formally, during per-instance training, we maximize the log likelihood under the constraints

$$
\begin{aligned}
\max_{\lambda, z} \ \ \  \mathbb{E}_{x_i}[\log p(x_i)] &= \mathbb{E}_{x_i}\left[-\sum_j \lambda_j z_j H_j(x_i; \theta_j^\star) - \log Z\right] \\
\text{subject to} \ \ \ \ \ \ \ \ \ \ \ \  \theta_j^\star &= \arg \min \ell_j(\{x_i\}, \theta_j), \quad \forall j,
\end{aligned}
$$

where the optimal filter parameters $\theta_j^\star$ best capture the hidden concepts modeled by the filter family of $H_j(\cdot; \theta_j)$. Such a bilevel design could improve filter expressiveness by finding the best filter in a continuous filter family while avoiding adding unnecessarily more filters.

After training a descriptive model for each instance, we use the model to solve a problem in two ways: Given a candidate set, we can pick the one with the highest/lowest probability as the answer; without one, we can perform maximum A posteriori (MAP) sampling from the distribution to render an answer with a rendering engine.
