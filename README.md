# LLMMathWriting

## Papers in this experiment

Please cite these papers if you are going to use the results from this repository.

```
@online{liNeuralSymbolicRecursiveMachineSystematicGeneralization2024,
  title       = {Neural-{{Symbolic Recursive Machine}} for {{Systematic Generalization}}},
  author      = {Li, Qing and Zhu, Yixin and Liang, Yitao and Wu, Ying Nian and Zhu, Song-Chun and Huang, Siyuan},
  date        = {2024-04-29},
  eprint      = {2210.01603},
  eprinttype  = {arxiv},
  eprintclass = {cs},
  doi         = {10.48550/arXiv.2210.01603},
  url         = {http://arxiv.org/abs/2210.01603},
  urldate     = {2024-05-01},
  abstract    = {Current learning models often struggle with human-like systematic generalization, particularly in learning compositional rules from limited data and extrapolating them to novel combinations. We introduce the Neural-Symbolic Recursive Machine (NSR), whose core is a Grounded Symbol System (GSS), allowing for the emergence of combinatorial syntax and semantics directly from training data. The NSR employs a modular design that integrates neural perception, syntactic parsing, and semantic reasoning. These components are synergistically trained through a novel deduction-abduction algorithm. Our findings demonstrate that NSR's design, imbued with the inductive biases of equivariance and compositionality, grants it the expressiveness to adeptly handle diverse sequence-to-sequence tasks and achieve unparalleled systematic generalization. We evaluate NSR's efficacy across four challenging benchmarks designed to probe systematic generalization capabilities: SCAN for semantic parsing, PCFG for string manipulation, HINT for arithmetic reasoning, and a compositional machine translation task. The results affirm NSR's superiority over contemporary neural and hybrid models in terms of generalization and transferability.},
  pubstate    = {preprint},
  keywords    = {Computer Science - Computation and Language,Computer Science - Computer Vision and Pattern Recognition,Computer Science - Machine Learning}
}
```

```
@online{yanOrchestrateLatentExpertiseAdvancingOnlineContinualLearningMultiLevelSupervisionReverseSelfDistillation2024,
  title       = {Orchestrate {{Latent Expertise}}: {{Advancing Online Continual Learning}} with {{Multi-Level Supervision}} and {{Reverse Self-Distillation}}},
  shorttitle  = {Orchestrate {{Latent Expertise}}},
  author      = {Yan, HongWei and Wang, Liyuan and Ma, Kaisheng and Zhong, Yi},
  date        = {2024-03-30},
  eprint      = {2404.00417},
  eprinttype  = {arxiv},
  eprintclass = {cs},
  url         = {http://arxiv.org/abs/2404.00417},
  urldate     = {2024-05-05},
  abstract    = {To accommodate real-world dynamics, artificial intelligence systems need to cope with sequentially arriving content in an online manner. Beyond regular Continual Learning (CL) attempting to address catastrophic forgetting with offline training of each task, Online Continual Learning (OCL) is a more challenging yet realistic setting that performs CL in a one-pass data stream. Current OCL methods primarily rely on memory replay of old training samples. However, a notable gap from CL to OCL stems from the additional overfitting-underfitting dilemma associated with the use of rehearsal buffers: the inadequate learning of new training samples (underfitting) and the repeated learning of a few old training samples (overfitting). To this end, we introduce a novel approach, Multi-level Online Sequential Experts (MOSE), which cultivates the model as stacked sub-experts, integrating multi-level supervision and reverse self-distillation. Supervision signals across multiple stages facilitate appropriate convergence of the new task while gathering various strengths from experts by knowledge distillation mitigates the performance decline of old tasks. MOSE demonstrates remarkable efficacy in learning new samples and preserving past knowledge through multi-level experts, thereby significantly advancing OCL performance over state-of-the-art baselines (e.g., up to 7.3\% on Split CIFAR-100 and 6.1\% on Split Tiny-ImageNet).},
  langid      = {english},
  pubstate    = {preprint},
  keywords    = {Computer Science - Artificial Intelligence,Computer Science - Computer Vision and Pattern Recognition,Computer Science - Machine Learning}
}
```

```
@article{zhangHumanlevelfewshotconceptinductionminimaxentropylearning2024,
  title        = {Human-Level Few-Shot Concept Induction through Minimax Entropy Learning},
  author       = {Zhang, Chi and Jia, Baoxiong and Zhu, Yixin and Zhu, Song-Chun},
  date         = {2024-04-19},
  journaltitle = {Science Advances},
  volume       = {10},
  number       = {16},
  pages        = {eadg2488},
  publisher    = {American Association for the Advancement of Science},
  doi          = {10.1126/sciadv.adg2488},
  url          = {https://www.science.org/doi/10.1126/sciadv.adg2488},
  urldate      = {2024-04-25},
  abstract     = {Humans learn concepts both from labeled supervision and by unsupervised observation of patterns, a process machines are being taught to mimic by training on large annotated datasets—a method quite different from the human pathway, wherein few examples with no supervision suffice to induce an unfamiliar relational concept. We introduce a computational model designed to emulate human inductive reasoning on abstract reasoning tasks, such as those in IQ tests, using a minimax entropy approach. This method combines identifying the most effective constraints on data via minimum entropy with determining the best combination of them via maximum entropy. Our model, which applies this unsupervised technique, induces concepts from just one instance, reaching human-level performance on tasks of Raven’s Progressive Matrices (RPM), Machine Number Sense (MNS), and Odd-One-Out (O3). These results demonstrate the potential of minimax entropy learning for enabling machines to learn relational concepts efficiently with minimal input.}
}
```

```
@inproceedings{yuDeepModularCoAttentionNetworksVisualQuestionAnswering2019,
  title      = {Deep {{Modular Co-Attention Networks}} for {{Visual Question Answering}}},
  booktitle  = {2019 {{IEEE}}/{{CVF Conference}} on {{Computer Vision}} and {{Pattern Recognition}} ({{CVPR}})},
  author     = {Yu, Zhou and Yu, Jun and Cui, Yuhao and Tao, Dacheng and Tian, Qi},
  date       = {2019-06},
  pages      = {6274--6283},
  publisher  = {IEEE},
  location   = {Long Beach, CA, USA},
  doi        = {10.1109/CVPR.2019.00644},
  url        = {https://ieeexplore.ieee.org/document/8953581/},
  urldate    = {2023-03-07},
  abstract   = {Visual Question Answering (VQA) requires a finegrained and simultaneous understanding of both the visual content of images and the textual content of questions. Therefore, designing an effective ‘co-attention’ model to associate key words in questions with key objects in images is central to VQA performance. So far, most successful attempts at co-attention learning have been achieved by using shallow models, and deep co-attention models show little improvement over their shallow counterparts. In this paper, we propose a deep Modular Co-Attention Network (MCAN) that consists of Modular Co-Attention (MCA) layers cascaded in depth. Each MCA layer models the self-attention of questions and images, as well as the question-guided-attention of images jointly using a modular composition of two basic attention units. We quantitatively and qualitatively evaluate MCAN on the benchmark VQA-v2 dataset and conduct extensive ablation studies to explore the reasons behind MCAN’s effectiveness. Experimental results demonstrate that MCAN significantly outperforms the previous state-of-the-art. Our best single model delivers 70.63\% overall accuracy on the test-dev set.},
  eventtitle = {2019 {{IEEE}}/{{CVF Conference}} on {{Computer Vision}} and {{Pattern Recognition}} ({{CVPR}})},
  isbn       = {978-1-72813-293-8},
  langid     = {english}
}
```

```
@inproceedings{choGenerativeBiasRobustVisualQuestionAnswering2023,
  title      = {Generative {{Bias}} for {{Robust Visual Question Answering}}},
  author     = {Cho, Jae Won and Kim, Dong-Jin and Ryu, Hyeonggon and Kweon, In So},
  date       = {2023},
  pages      = {11681--11690},
  url        = {https://openaccess.thecvf.com/content/CVPR2023/html/Cho_Generative_Bias_for_Robust_Visual_Question_Answering_CVPR_2023_paper.html},
  urldate    = {2024-04-13},
  eventtitle = {Proceedings of the {{IEEE}}/{{CVF Conference}} on {{Computer Vision}} and {{Pattern Recognition}}},
  langid     = {english}
}
```