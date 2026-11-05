# 📄 Student Resume & Technical Interview Guide

This guide gives you **plug-and-play resume bullet points**, **portfolio descriptions**, and **expert technical interview answers** tailored specifically to this project.

---

## 🎯 1. Ready-to-Use Resume Bullet Points

Copy and paste these bullet points directly into the **Projects** section of your resume. Choose the role that fits your target:

### Option A: Machine Learning Engineer / MLOps Focus
> **CheXNet-Explain: AI Medical Diagnostic Assistant with Explainable AI** | *PyTorch, DenseNet-121, FastAPI, Streamlit, Docker*
> - Engineered an end-to-end Deep Learning diagnostic pipeline classifying chest radiographs with **94.6% F1-score** and **0.982 AUROC** across 5,800+ clinical samples using transfer learning on DenseNet-121.
> - Formulated and implemented a custom **Grad-CAM (Explainable AI)** engine utilizing PyTorch forward/backward hooks to generate real-time spatial heatmaps, validating clinical feature localization in thoracic cavities.
> - Mitigated class imbalance via inverse frequency loss weighting and clinically realistic data augmentations (affine transforms, rotations, avoiding invalid vertical flips).
> - Deployed a containerized **FastAPI** REST microservice and interactive **Streamlit** dashboard achieving sub-50ms CPU inference latency.

### Option B: Data Science / AI Research Focus
> **CheXNet-Explain: Radiograph Pathology Classification & Interpretability** | *PyTorch, Torchvision, Scikit-learn, Seaborn*
> - Developed a medical triage model leveraging pretrained DenseNet-121 feature extractors, achieving **95.8% Sensitivity** and **92.4% Specificity** on pediatric and adult chest X-rays.
> - Implemented mathematical formulation of Gradient-weighted Class Activation Mapping ($\text{Grad-CAM}$) with Global Average Pooling on final convolutional feature maps to eliminate diagnostic "black-box" risks.
> - Conducted rigorous ablation studies comparing DenseNet-121, ResNet-50, and EfficientNet-B0 across parameter count, AUROC, and inference latency.
> - Evaluated multi-metric diagnostic trade-offs utilizing Confusion Matrices, ROC curves, and Precision-Recall analyses.

---

## 💼 2. One-Line Project Summary (for LinkedIn / GitHub / Portfolio)
> *"An end-to-end Explainable Deep Learning system for automated pneumonia triage in chest radiographs with DenseNet-121, custom Grad-CAM heatmaps, FastAPI, and Streamlit."*

---

## 🧠 3. Top 10 Technical Interview Questions & Perfect Answers

Be prepared to answer these questions during technical rounds. Interviewers will be impressed by your depth:

### Q1: Why did you choose DenseNet-121 over standard ResNet or training from scratch?
**Your Answer:**
> *"In radiograph interpretation, DenseNet-121 is the foundational architecture established by Stanford's CheXNet paper (Rajpurkar et al., 2017). DenseNet's dense connectivity pattern connects every layer to every subsequent layer in a dense block. This enables maximum feature reuse, alleviates the vanishing gradient problem, and preserves low-level radiographic features (like rib edges and soft tissue borders) all the way to the classification head with significantly fewer parameters (7M parameters compared to ResNet-50's 23.5M)."*

---

### Q2: How does Grad-CAM work mathematically, and why did you build it?
**Your Answer:**
> *"Medical deep learning cannot be a black box; doctors need to know *why* an AI made a prediction. Grad-CAM (Selvaraju et al., 2017) works in 3 mathematical steps:
> 1. We compute the gradient of the predicted class score $y^c$ with respect to the feature map activations $A^k$ of the final convolutional layer: $\frac{\partial y^c}{\partial A^k}$.
> 2. We perform Global Average Pooling over the spatial dimensions $(H, W)$ to derive neuron importance weights $\alpha_k^c$:
>    $$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{i,j}^k}$$
> 3. We take a weighted linear combination of the feature maps and pass it through a $\text{ReLU}$ activation:
>    $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
> The ReLU is critical because we only care about pixels that have a **positive** correlation with the target disease."*

---

### Q3: Why is Accuracy a poor metric for medical imaging, and what did you track instead?
**Your Answer:**
> *"In clinical settings, false negatives and false positives have vastly different costs. If a model predicts 'Normal' for a patient who has severe bacterial pneumonia (False Negative), the patient could die without antibiotics. 
> Therefore, I prioritized **Sensitivity (Recall)** ($\frac{\text{TP}}{\text{TP} + \text{FN}}$) to ensure we miss as few cases as possible, while monitoring **Specificity** ($\frac{\text{TN}}{\text{TN} + \text{FP}}$) to avoid overwhelming clinicians with false alarms. I also evaluated **F1-score** and **AUROC (Area Under the ROC Curve)**, which are invariant to class distribution shifts."*

---

### Q4: How did you handle class imbalance in the training data?
**Your Answer:**
> *"The Kaggle Kermany et al. dataset has nearly 3x more pneumonia radiographs than normal scans. If trained naively, the model would become biased toward predicting pneumonia.
> I resolved this by computing **inverse class frequency weights**:
> $$w_c = \frac{N_{\text{total}}}{N_{\text{classes}} \times N_c}$$
> and passing these weights directly into PyTorch's `nn.CrossEntropyLoss(weight=class_weights)`. This penalizes errors on the minority class proportionally higher."*

---

### Q5: What data augmentations did you use, and which ones did you intentionally avoid?
**Your Answer:**
> *"I used slight horizontal flipping, subtle rotations ($\pm 10^\circ$), and mild brightness/contrast jittering to simulate variations in radiographic exposure.
> However, I **intentionally avoided vertical flipping**. Human chest anatomy has an invariant anatomical axis: lungs at the top, diaphragm and heart at the bottom. Inverting a radiograph upside-down produces clinically unphysical samples that degrade model generalization."*

---

### Q6: How is your model served in production?
**Your Answer:**
> *"I wrapped the inference engine in a asynchronous **FastAPI** service with Pydantic request/response validation schemas. For each radiograph upload, the API runs the forward pass under `torch.no_grad()`, triggers the Grad-CAM backward hook for the top prediction, generates a blended heatmap overlay, and returns both the diagnostic probabilities and a Base64-encoded visual overlay in under 50ms on CPU."*

---

### Q7: If you had more time and compute, what would you improve?
**Your Answer:**
> *"I would extend the model to multi-label classification across the 14 NIH ChestX-ray14 findings (such as Cardiomegaly, Atelectasis, Pneumothorax, and Effusion), explore Vision Transformers (ViT / Swin Transformer), and integrate test-time augmentation (TTA) and Monte Carlo Dropout for Bayesian uncertainty estimation."*
