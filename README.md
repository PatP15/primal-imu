# PRIMAL: Physically Reactive and Interactive Motor Model for Avatar Learning

This repo is developed based on pytorch-lightning, hydra, huggingface libs, and others.

[**project page**](https://yz-cnsdqz.github.io/eigenmotion/PRIMAL/)

## Documentation

The documentation has been moved to the `docs/` directory to keep the project organized.

*   **[Installation Guide](docs/installation.md)**: Setting up the environment (conda/poetry) and data.
*   **[Unreal Engine Demo](docs/installation.md)**: Instructions for the UE demo backend (see Installation).
*   **[Experiment Guide](docs/experiments.md)**: Detailed instructions for running training, inference, and custom experiments.
*   **[SLURM Cluster Guide](docs/slurm.md)**: Instructions for running jobs on SLURM clusters.
*   **[Acceleration Model Guide](docs/acceleration_model.md)**: Specifics about the acceleration-based model variant.

## Quick Start

### 1. Inference with Gradio

Run the gradio demo for the base model:
```bash
poetry run python demos/ARDiffusion_gradio.py logs/motion_diffuser_ar/runs/silu_anchor
```

For the action generation model:
```bash
poetry run python demos/ARDiffusionAction_gradio.py logs/motion_diffuser_ar_action/runs/ours
```

### 2. Training

To pretrain the base model:
```bash
python scripts/train.py --config-name=train_diffusion task_name=motion_diffuser_ar
```

For detailed training instructions, including adaptation and custom datasets, please refer to the **[Experiment Guide](docs/experiments.md)**.

## License and Citation

See [license file](LICENSE) for more details. Please cite the following work if it helps.

```bibtex
@inproceedings{primal:iccv:2025,
  author = {Zhang, Yan and Feng, Yao and Cseke, Alpár and Saini, Nitin and Bajandas, Nathan and Heron, Nicolas and Black, Michael J.},  
   title = {{PRIMAL:} Physically Reactive and Interactive Motor Model for Avatar Learning},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month = oct,
  year = {2025}
}
```
