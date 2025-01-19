# Towards Continuous Resolution Video Prediction

This repository contains all the code associated with my master's thesis, **"Towards Continuous Resolution Video Prediction"**.

---

## 📁 Repository Structure

- **`scripts/`**: Python scripts for dataset generation and training models.
- **`configs/`**: Configuration files used for training and experiments.
  - Configs with the `FINAL` prefix were used for the final experiments.

---

## 🚀 Getting Started

### Environment Setup
To run the code locally, set up the environment using Task and Micromamba:

```bash
task env-create
micromamba activate vitssm
```

Alternatively, you can run the experiments on Google Colab using the provided **`colab_runner.ipynb`** notebook.

---

### Example Usage
Train a model using the following command:

```bash
python scripts/train_upt_vae_cont.py --config UPT/FINAL_upt_vae_t_cont_config.yml
```

---

## 📊 Resources

- **Dataset**: [Download Dataset](https://drive.google.com/drive/folders/14HFpG_VFwsD-uI9br4A2EA9LrQKP9SUO?usp=sharing)  
  The dataset used for the experiments is available here.

- **Model Checkpoints**: [Download Checkpoints](https://drive.google.com/drive/folders/1Ut7K4dOtpe5xk8VmRYcHzpZYYHPjaawg?usp=sharing)  
  Pretrained model checkpoints can be accessed here.

---

## 📝 Notes
- All experiments and results presented in the thesis used configurations prefixed with `FINAL`.

---

## 📫 Contact
For any questions or discussions, feel free to reach out!
