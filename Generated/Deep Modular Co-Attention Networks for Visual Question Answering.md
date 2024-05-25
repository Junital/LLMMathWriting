In this section, we explore the Deep Modular Co-Attention Network designed for the Visual Question Answering (VQA) task. The architecture involves two primary attention units: Self-Attention (SA) and Guided-Attention (GA). These units work together to deepen the co-attention network, thereby enhancing the model's ability to learn representations and patterns effectively.

### Self-Attention (SA) Unit

The Self-Attention unit aims to learn the relationships within a single modality. Given the input features $X \in \mathbb{R}^{m \times d_x}$, where $X = [x_1; \dots; x_m]$ represents $m$ feature vectors each of dimension $d_x$, the SA unit processes these features to produce output features $Z \in \mathbb{R}^{m \times d}$.

### Guided-Attention (GA) Unit

The Guided-Attention unit focuses on learning the relationships between different modalities. It takes as input both the features $X$ and another set of features $Y \in \mathbb{R}^{n \times d_y}$, where $Y = [y_1; \dots; y_n]$ consists of $n$ feature vectors each of dimension $d_y$. The GA unit guides the features in $X$ using the features in $Y$ to produce the output features $Z \in \mathbb{R}^{m \times d}$.

The two attention units can be formally described as follows:

1. **Self-Attention (SA) Unit**:
   - Input: $X = [x_1; \dots; x_m] \in \mathbb{R}^{m \times d_x}$
   - Output: $Z \in \mathbb{R}^{m \times d}$

2. **Guided-Attention (GA) Unit**:
   - Inputs: $X = [x_1; \dots; x_m] \in \mathbb{R}^{m \times d_x}$ and $Y = [y_1; \dots; y_n] \in \mathbb{R}^{n \times d_y}$
   - Output: $Z \in \mathbb{R}^{m \times d}$

These attention mechanisms work in tandem to allow the network to focus on relevant parts of the input features, whether they are within the same modality (SA) or across different modalities (GA). As such, the deep modular co-attention network is able to capture intricate patterns and dependencies that are essential for accurately answering visual questions.

In summary, the incorporation of both self-attention and guided-attention units into the co-attention network facilitates a more comprehensive and detailed learning process, thereby enabling the model to perform better on the VQA task.
