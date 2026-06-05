# Comparative Study of YOLO, Transformer and Mask R-CNN for Instance Segmentation of Retinal Fundus Images with Explainable AI Techniques

## Abstract
This work presents an instance segmentation pipeline for retinal fundus images, integrating **YOLOv8/11‑seg** (object detection + segmentation), **SAM2** (promptable segmentation refinement), and **HiResCAM** (explanatory visualisation). The system achieves precise localisation and delineation of retinal lesions, with the proposed architecture validated through comparative performance analysis. A live interactive demo is provided, along with trained model weights for reproducibility.

## 1. Methodology

The proposed framework (Figure 1) combines the efficiency of YOLOv8/11‑seg for initial lesion proposal, the generalisation capability of SAM2 for mask refinement, and HiResCAM for generating class‑activation maps that highlight diagnostically relevant regions.

**Figure 1 – Proposed YOLO‑HiResCAM Architecture**  
![Proposed YOLO‑HiResCAM Architecture](https://github.com/Addisu-Amare/Res/blob/main/architecture-diagram-yolo-v-8.png)

## 2. Segmentation Results

A representative segmentation output is shown in Figure 2. The combination of YOLO‑seg and SAM2 produces smooth, instance‑level lesion masks directly overlaid on the fundus image.

**Figure 2 – Example Segmentation Output (AVIF format)**  
![Segmentation Result](https://raw.githubusercontent.com/Addisu-Amare/Res/main/avif-result.avif)

**Figure 3 – YOLO‑SAM‑XAI Integration**  
![YOLO‑SAM‑XAI](https://github.com/Addisu-Amare/Res/blob/main/yolo-sam-arch.png)

## 3. Comparative Evaluation

Quantitative and qualitative comparisons against baseline models are summarised in Figure 4. The proposed method demonstrates improved segmentation fidelity and explanatory clarity.

**Figure 4 – Performance Comparison**  
![Comparison Result](https://github.com/Addisu-Amare/Res/blob/main/comparison-graph_1.png)

## 4. Live Demonstration

An interactive Space is hosted on Hugging Face, allowing real‑time testing on custom fundus images.

[![Open in Hugging Face Spaces](https://img.shields.io/badge/Open%20in-Spaces-blue?logo=huggingface)](https://huggingface.co/spaces/woldemerkorios/Retina_lesion)

**Access the live demo:** [Retina Lesion Space](https://huggingface.co/spaces/woldemerkorios/Retina_lesion)

## 5. Model Availability

Trained weights for the YOLOv8‑seg variants are provided for reproducibility and further fine‑tuning.

- [yolov8n_seg_100epoch_best.pt](https://huggingface.co/spaces/woldemerkorios/Retina_lesion/blob/main/yolov8n_seg_16_batch_100epoch_best.pt)  
  *Nano variant – 16 batch size, 100 epochs*
- [yolov8s_seg_100epoch_best.pt](https://huggingface.co/spaces/woldemerkorios/Retina_lesion/blob/main/yolov8s_seg_16batch_100epoch_best.pt)  
  *Small variant – 16 batch size, 100 epochs*

  ##  Tackling  instance segmentation  problem of missing lession  using:Weighted Mask Ensemble + Mask NMS
  -yolov8n-seg+yolov8s-seg: weighted ensemble
  - Here is the demonstration image:
  - 

## References

- Ultralytics YOLOv8‑seg: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)  
- SAM2 (Meta): [https://github.com/facebookresearch/sam2](https://github.com/facebookresearch/sam2)  
- HiResCAM (Explainable AI): [https://arxiv.org/abs/2011.08891](https://arxiv.org/abs/2011.08891)
