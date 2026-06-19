# Citation Audit

This audit checks that protocol citations refer to real papers and that the protocol uses them in the correct context.

## Zotero Coverage

Inspection and live Zotero bridge verification found:

- Existing local collection: `Neurogenesis Deeplearning` in `My Library`, 18 items. It contains several relevant continual-learning papers but uses generic citekeys such as `2016`, `2017b`, and `2019`.
- Existing group library: `Routing` originally contained 19 of the 20 protocol literature keys with exact citekeys. `maile2022north` and `yoon2018den` were imported so `Routing` now verifies with all 20 protocol citekeys.
- Target group library: `Neurogenesis` originally contained Draelos et al. and DEN with the protocol keys. The remaining protocol entries were imported through the Zotero bridge, and a final dry-run verified all 20 required citekeys in `Neurogenesis`.

Zotero's local API is available at `127.0.0.1:23119` after enabling the connector HTTP server through the persistent profile `user.js` override.

## Entry-Level Audit

| Citekey | Zotero status | Protocol context | Audit result |
|---|---:|---|---|
| `draelos2017neurogenesis` | Present in `Neurogenesis` and `Routing` | Original NDL method, reconstruction-error trigger, new-class setting | Appropriate. Metadata corrected to IJCNN 2017 with DOI `10.1109/IJCNN.2017.7965898`. |
| `fahlman1990cascade` | Present in `Routing` and `Neurogenesis` | Constructive networks add hidden units when residual error/stagnation remains | Appropriate. |
| `rusu2016progressive` | Present in `My Library`, `Routing`, and `Neurogenesis` | Progressive columns freeze old task capacity and add task-specific capacity | Appropriate. |
| `yoon2018den` | Present in `Neurogenesis`, `Routing`, and `My Library` under different citekeys | Dynamically Expandable Networks expand capacity for lifelong learning | Appropriate. |
| `yan2021der` | Present in `Routing` and `Neurogenesis` | Dynamically Expandable Representation for class-incremental learning | Appropriate. Metadata enriched with CVPR DOI. |
| `fernando2017pathnet` | Present in `Routing` and `Neurogenesis` | Modular path/routing approach for continual learning | Appropriate. |
| `li2019learn` | Present in `My Library`, `Routing`, and `Neurogenesis` | Learns sharing/adaptation/expansion structure for continual learning | Appropriate. Metadata enriched with PMLR pages. |
| `han2015deep` | Present in `Routing` and `Neurogenesis` | Magnitude-style pruning/removing unimportant connections | Appropriate. |
| `guo2016dynamic` | Present in `Routing` and `Neurogenesis` | Pruning plus splicing/restoration after mistaken removals | Appropriate. |
| `liu2017network` | Present in `Routing` and `Neurogenesis` | Channel-level sparsity and structured channel pruning | Appropriate. Metadata enriched with ICCV DOI/pages. |
| `louizos2018learning` | Present in `Routing` and `Neurogenesis` | L0 gates/regularization induce sparsity during training | Appropriate. |
| `dai2018nest` | Present in `Routing` and `Neurogenesis` | Grow-and-prune architecture synthesis | Appropriate. |
| `gordon2018morphnet` | Present in `Routing` and `Neurogenesis` | Resource-aware structure learning that can shrink/expand networks | Appropriate. Metadata enriched with CVPR DOI/pages. |
| `mocanu2018scalable` | Present in `Routing` and `Neurogenesis` | Sparse Evolutionary Training with adaptive sparse connectivity | Appropriate. DOI present. |
| `evci2020rigging` | Present in `Routing` and `Neurogenesis` | RigL regrows sparse connections using gradient information | Appropriate. Metadata enriched with PMLR pages. |
| `chen2016net2net` | Present in `Routing` and `Neurogenesis` | Function-preserving widening/deepening transformations | Appropriate. |
| `wei2016network` | Present in `Routing` and `Neurogenesis` | Network morphism preserves function while changing depth/width/kernel/subnet | Appropriate. Metadata enriched with PMLR volume. |
| `liu2019darts` | Present in `Routing` and `Neurogenesis` | Differentiable architecture search over architecture parameters | Appropriate. |
| `evci2022gradmax` | Present in `Routing` and `Neurogenesis` | Gradient-maximizing neuron growth | Appropriate. Metadata corrected in `references.bib` with full authors. |
| `maile2022north` | Present in `Routing` and `Neurogenesis` | NORTH* orthogonality-based neurogenesis trigger/initialization strategies | Appropriate. Imported after local API was restored. |

## Required Follow-Up

No bibliography completeness follow-up remains for the protocol citekeys. If the bibliography is later regenerated from Better BibTeX, export the complete `Neurogenesis` group and keep the existing citekeys stable.
