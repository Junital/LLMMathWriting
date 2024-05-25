In order for our bias model GenB to learn the biases given the question, we use the traditional VQA loss, the Binary Cross Entropy Loss:

$$\mathcal{L}_{GT}(F_{b, G}) = \mathcal{L}_{BCE}(\sigma(F_{b, G}(\mathbf{z}, \mathbf{q})), \mathbf{y}_{gt}),
$$

However, unlike existing works, we want the bias model to also capture the biases in the target model. Hence, in order to mimic the bias of the target model as a random distribution of the answers, we propose adversarial training [16] similar to [29] to train our bias model. In particular, we introduce a discriminator that distinguishes the answers from the target model and the bias model as “real” and “fake” answers respectively. The discriminator is formulated as $D(F(\mathbf{v}, \mathbf{q}))$ and $D(F_{b,G}(\mathbf{v}, \mathbf{q}))$ or rewritten as $D(\mathbf{y})$ and $D(\mathbf{y}_b)$. he objective of our adversarial network with generator $F_{b,G}(\cdot, \cdot)$ and $D(\cdot)$ can be expressed as follows:

$$
\mathcal{L}_{GAN}(F_{b, G}, D) = \mathbb{E}_{\mathbf{y}}\left [ \log \left ( D\left (\mathbf{y}\right )\right )\right ] + \mathbb{E}_{\mathbf{y}_b}\left [ \log \left (1- D\left (\mathbf{y}_b\right )\right )\right ],
$$

The generator ($F_{b,G}$) tries to minimize the objective ($\mathcal{L}_{GAN}$) against an adversarial discriminator ($D$) that tries to maximize it. Through alternative training of $D$ and $F_{b,G}$, he distribution of the answer vector from the bias model ($\mathbf{y}_b$) should be close to that from the target model ($\mathbf{y}$).

In addition, to further enforce bias model to capture the intricate biases present in the target model, we add an additional knowledge distillation objective [20] similar to [12, 27, 30, 32, 33] to ensure that the model bias model is able to follow the behavior of the target model with only the $\mathbf{q}$ given to it. We empirically find that it is beneficial to include a sample-wise distance based metric such as KL divergence. This method is similar to the approaches in the image to image translation task [21]. Then, the goal of the generator is not only to fool the discriminator but also to try to imitate the answer output of the target model in order to give the target model more challenging supervision in the form of hard negative sample synthesis. We add another objective to our adversarial training for $F_{b,G}(\cdot, \cdot)$:

$$
\mathcal{L}_{distill}(F_{b, G}) = \mathbb{E}_{v, q, z}\left [ D_{KL} \left (F(\mathbf{v}, \mathbf{q}) || F_{b, G}(\mathbf{z}, \mathbf{q})\right )\right ],
$$

Ultimately, the final training loss for the bias model, or GenB, is as follows:

$$
\mathcal{L}_{GenB}(F_{b, G}, D) = \mathcal{L}_{GAN}(F_{b, G}, D) + \lambda_1 \mathcal{L}_{distill}(F_{b, G}) + \lambda_2 \mathcal{L}_{GT}(F_{b, G}).
$$

where $\lambda_1$ and $\lambda_2$ are the loss weight hyper-parameters.
