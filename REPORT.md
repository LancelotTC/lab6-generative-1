# Configuration Report

This document records configurations that have been executed, based on run artifacts in `runs/`.

## Dataset Choice

I chose the `Hand` MedNIST subdataset for the main experiments because it is the easiest class to visually assess when checking whether generated images are correct or not.

Out of curiosity, I also ran the current best-performing model on the `CXR` MedNIST subdataset to evaluate transfer of performance. The CXR-specific results will be added once they are available.

## Hardware Specification

- CPU: AMD Ryzen 5 7600
- GPU: NVIDIA GeForce RTX 4060 Ti (16 GB VRAM)
- System Memory: G.Skill 32 GB DDR5 RAM
- Storage: Western Digital SN770

## Data Source

- `runs/*/*/hyperparameters.json`
- `runs/*/*/metrics.json` (when present)

## Summary

- GAN runs found: 5
- LDM runs found: 5
- Completed runs (metrics present): 9
- Incomplete runs (metrics missing): 1

## GAN Runs

### Run `20260322_174818_656884`

| Metric                         | Value        |
| ------------------------------ | ------------ |
| Status                         | completed    |
| Epochs                         | 60           |
| Batch Size                     | 64           |
| Train/Valid Ratio              | 0.8          |
| AE Channels                    | [16, 24, 32] |
| Latent Channels                | 16           |
| Residual Blocks                | 2            |
| Discriminator Layers           | 3            |
| Discriminator Channels         | 16           |
| Learning Rate (Generator)      | 0.0001       |
| Learning Rate (Discriminator)  | 0.0005       |
| KL Weight                      | 1e-06        |
| Perceptual Weight              | 0.001        |
| Adversarial Weight             | 0.01         |
| Best Validation Reconstruction | 0.0247379    |
| Test Reconstruction            | 0.0258646    |
| Run Duration (s)               | 1294.23      |

<table cellspacing="18" cellpadding="6">
  <tr>
    <td align="center" valign="top" width="33%">
      <img src="runs/GAN/20260322_174818_656884/plots/GAN%20-%20Adversarial%20Training%20Curves.png" width="300" />
    </td>
    <td align="center" valign="top" width="33%">
      <img src="runs/GAN/20260322_174818_656884/plots/GAN%20-%20Latent%20Space%20(t-SNE%202D).png" width="300" />
    </td>
    <td align="center" valign="top" width="33%">
      <img src="runs/GAN/20260322_174818_656884/plots/GAN%20-%20Latent%20Interpolation.gif" width="300" />
    </td>
  </tr>
  <tr>
    <td align="center" valign="top" width="33%">
      <sub>GAN - Adversarial Training Curves</sub>
    </td>
    <td align="center" valign="top" width="33%">
      <sub>GAN - Latent Space (t-SNE 2D)</sub>
    </td>
    <td align="center" valign="top" width="33%">
      <sub>GAN - Latent Interpolation.gif</sub>
    </td>
  </tr>
</table>

### Run `20260322_205424_037424`

| Metric                         | Value        |
| ------------------------------ | ------------ |
| Status                         | completed    |
| Epochs                         | 60           |
| Batch Size                     | 32           |
| Train/Valid Ratio              | 0.95         |
| AE Channels                    | [16, 24, 32] |
| Latent Channels                | 16           |
| Residual Blocks                | 2            |
| Discriminator Layers           | 3            |
| Discriminator Channels         | 16           |
| Learning Rate (Generator)      | 0.0001       |
| Learning Rate (Discriminator)  | 0.0005       |
| KL Weight                      | 1e-06        |
| Perceptual Weight              | 0.001        |
| Adversarial Weight             | 0.01         |
| Best Validation Reconstruction | 0.0211905    |
| Test Reconstruction            | 0.022213     |
| Run Duration (s)               | 1198.67      |

<table cellspacing="18" cellpadding="6">
  <tr>
    <td align="center" valign="top" width="33%">
      <img src="runs/GAN/20260322_205424_037424/plots/GAN%20-%20Adversarial%20Training%20Curves.png" width="300" />
    </td>
    <td align="center" valign="top" width="33%">
      <img src="runs/GAN/20260322_205424_037424/plots/GAN%20-%20Latent%20Space%20(t-SNE%202D).png" width="300" />
    </td>
    <td align="center" valign="top" width="33%">
      <img src="runs/GAN/20260322_205424_037424/plots/GAN%20-%20Latent%20Interpolation.gif" width="300" />
    </td>
  </tr>
  <tr>
    <td align="center" valign="top" width="33%">
      <sub>GAN - Adversarial Training Curves</sub>
    </td>
    <td align="center" valign="top" width="33%">
      <sub>GAN - Latent Space (t-SNE 2D)</sub>
    </td>
    <td align="center" valign="top" width="33%">
      <sub>GAN - Latent Interpolation.gif</sub>
    </td>
  </tr>
</table>

### Run `20260323_022333_502406`

| Metric                         | Value        |
| ------------------------------ | ------------ |
| Status                         | completed    |
| Epochs                         | 60           |
| Batch Size                     | 64           |
| Train/Valid Ratio              | 0.95         |
| AE Channels                    | [24, 48, 64] |
| Latent Channels                | 12           |
| Residual Blocks                | 3            |
| Discriminator Layers           | 3            |
| Discriminator Channels         | 16           |
| Learning Rate (Generator)      | 0.0001       |
| Learning Rate (Discriminator)  | 0.0005       |
| KL Weight                      | 1e-06        |
| Perceptual Weight              | 0.01         |
| Adversarial Weight             | 0.01         |
| Best Validation Reconstruction | 0.0193268    |
| Test Reconstruction            | 0.0197533    |
| Run Duration (s)               | 1945.94      |

<table cellspacing="18" cellpadding="6">
  <tr>
    <td align="center" valign="top" width="33%">
      <img src="runs/GAN/20260323_022333_502406/plots/GAN%20-%20Adversarial%20Training%20Curves.png" width="300" />
    </td>
    <td align="center" valign="top" width="33%">
      <img src="runs/GAN/20260323_022333_502406/plots/GAN%20-%20Latent%20Space%20(t-SNE%202D).png" width="300" />
    </td>
    <td align="center" valign="top" width="33%">
      <img src="runs/GAN/20260323_022333_502406/plots/GAN%20-%20Latent%20Interpolation.gif" width="300" />
    </td>
  </tr>
  <tr>
    <td align="center" valign="top" width="33%">
      <sub>GAN - Adversarial Training Curves</sub>
    </td>
    <td align="center" valign="top" width="33%">
      <sub>GAN - Latent Space (t-SNE 2D)</sub>
    </td>
    <td align="center" valign="top" width="33%">
      <sub>GAN - Latent Interpolation.gif</sub>
    </td>
  </tr>
</table>

### Run `20260323_102522_535736`

| Metric                         | Value        |
| ------------------------------ | ------------ |
| Status                         | incomplete   |
| Epochs                         | 60           |
| Batch Size                     | 64           |
| Train/Valid Ratio              | 0.95         |
| AE Channels                    | [16, 24, 32] |
| Latent Channels                | 24           |
| Residual Blocks                | 3            |
| Discriminator Layers           | 3            |
| Discriminator Channels         | 16           |
| Learning Rate (Generator)      | 0.0001       |
| Learning Rate (Discriminator)  | 0.0005       |
| KL Weight                      | 1e-05        |
| Perceptual Weight              | 0.001        |
| Adversarial Weight             | 0.01         |
| Best Validation Reconstruction | -            |
| Test Reconstruction            | -            |
| Run Duration (s)               | -            |

<table cellspacing="18" cellpadding="6">
  <tr>
    <td align="center" valign="top" width="33%">
      <sub>Not available</sub>
    </td>
    <td align="center" valign="top" width="33%">
      <sub>Not available</sub>
    </td>
    <td align="center" valign="top" width="33%">
      <sub>Not available</sub>
    </td>
  </tr>
  <tr>
    <td align="center" valign="top" width="33%">
      <sub>GAN - Adversarial Training Curves</sub>
    </td>
    <td align="center" valign="top" width="33%">
      <sub>GAN - Latent Space (t-SNE 2D)</sub>
    </td>
    <td align="center" valign="top" width="33%">
      <sub>GAN - Latent Interpolation.gif</sub>
    </td>
  </tr>
</table>

### Run `20260323_103245_605233`

| Metric                         | Value        |
| ------------------------------ | ------------ |
| Status                         | completed    |
| Epochs                         | 60           |
| Batch Size                     | 64           |
| Train/Valid Ratio              | 0.95         |
| AE Channels                    | [16, 24, 32] |
| Latent Channels                | 24           |
| Residual Blocks                | 3            |
| Discriminator Layers           | 3            |
| Discriminator Channels         | 16           |
| Learning Rate (Generator)      | 0.0001       |
| Learning Rate (Discriminator)  | 0.0005       |
| KL Weight                      | 1e-05        |
| Perceptual Weight              | 0.001        |
| Adversarial Weight             | 0.01         |
| Best Validation Reconstruction | 0.0283249    |
| Test Reconstruction            | 0.0277679    |
| Run Duration (s)               | 1255.39      |

<table cellspacing="18" cellpadding="6">
  <tr>
    <td align="center" valign="top" width="33%">
      <img src="runs/GAN/20260323_103245_605233/plots/GAN%20-%20Adversarial%20Training%20Curves.png" width="300" />
    </td>
    <td align="center" valign="top" width="33%">
      <img src="runs/GAN/20260323_103245_605233/plots/GAN%20-%20Latent%20Space%20(t-SNE%202D).png" width="300" />
    </td>
    <td align="center" valign="top" width="33%">
      <img src="runs/GAN/20260323_103245_605233/plots/GAN%20-%20Latent%20Interpolation.gif" width="300" />
    </td>
  </tr>
  <tr>
    <td align="center" valign="top" width="33%">
      <sub>GAN - Adversarial Training Curves</sub>
    </td>
    <td align="center" valign="top" width="33%">
      <sub>GAN - Latent Space (t-SNE 2D)</sub>
    </td>
    <td align="center" valign="top" width="33%">
      <sub>GAN - Latent Interpolation.gif</sub>
    </td>
  </tr>
</table>

## LDM Runs

### Run `20260322_183659_301443`

| Metric                             | Value        |
| ---------------------------------- | ------------ |
| Status                             | completed    |
| Batch Size                         | 64           |
| Train/Valid Ratio                  | 0.8          |
| Autoencoder Epochs                 | 60           |
| Diffusion Epochs                   | 80           |
| AE Channels                        | [16, 24, 32] |
| Latent Channels                    | 16           |
| Residual Blocks                    | 2            |
| Discriminator Layers               | 3            |
| Discriminator Channels             | 16           |
| Diffusion Channels                 | [16, 24, 32] |
| Diffusion Head Channels            | [0, 4, 8]    |
| Diffusion Timesteps                | 1000         |
| Diffusion Beta Start               | 0.0015       |
| Diffusion Beta End                 | 0.0195       |
| Last Validation Loss (Autoencoder) | 0.0254651    |
| Last Validation Loss (Diffusion)   | 0.214465     |
| Run Duration (s)                   | 2997.81      |

<table cellspacing="18" cellpadding="6">
  <tr>
    <td align="center" valign="top" width="33%">
      <sub>Not available</sub>
    </td>
    <td align="center" valign="top" width="33%">
      <img src="runs/LDM/20260322_183659_301443/plots/Diffusion%20Model%20-%20Latent%20Space%20(t-SNE%202D).png" width="300" />
    </td>
    <td align="center" valign="top" width="33%">
      <img src="runs/LDM/20260322_183659_301443/plots/MedNIST%20Interpolation.gif" width="300" />
    </td>
  </tr>
  <tr>
    <td align="center" valign="top" width="33%">
      <sub>Diffusion Model - Adversarial Training Curves</sub>
    </td>
    <td align="center" valign="top" width="33%">
      <sub>Diffusion Model - Latent Space (t-SNE 2D)</sub>
    </td>
    <td align="center" valign="top" width="33%">
      <sub>MedNIST Interpolation.gif</sub>
    </td>
  </tr>
</table>

<table cellspacing="18" cellpadding="6">
  <tr>
    <td align="center" valign="top" colspan="3">
      <img src="runs/LDM/20260322_183659_301443/plots/Diffusion%20Model%20-%20Decoded%20Intermediates%20Every%20100%20Steps.png" width="972" />
    </td>
  </tr>
  <tr>
    <td align="center" valign="top" colspan="3">
      <sub>Diffusion Model - Decoded Intermediates Every 100 Steps</sub>
    </td>
  </tr>
</table>

### Run `20260322_211427_053137`

| Metric                             | Value        |
| ---------------------------------- | ------------ |
| Status                             | completed    |
| Batch Size                         | 32           |
| Train/Valid Ratio                  | 0.95         |
| Autoencoder Epochs                 | 60           |
| Diffusion Epochs                   | 80           |
| AE Channels                        | [16, 24, 32] |
| Latent Channels                    | 16           |
| Residual Blocks                    | 2            |
| Discriminator Layers               | 3            |
| Discriminator Channels             | 16           |
| Diffusion Channels                 | [4, 8, 16]   |
| Diffusion Head Channels            | [0, 4, 8]    |
| Diffusion Timesteps                | 1200         |
| Diffusion Beta Start               | 0.0015       |
| Diffusion Beta End                 | 0.0195       |
| Last Validation Loss (Autoencoder) | 0.022523     |
| Last Validation Loss (Diffusion)   | 0.786427     |
| Run Duration (s)                   | 4062.66      |

<table cellspacing="18" cellpadding="6">
  <tr>
    <td align="center" valign="top" width="33%">
      <img src="runs/LDM/20260322_211427_053137/plots/Diffusion%20Model%20-%20Adversarial%20Training%20Curves.png" width="300" />
    </td>
    <td align="center" valign="top" width="33%">
      <img src="runs/LDM/20260322_211427_053137/plots/Diffusion%20Model%20-%20Latent%20Space%20(t-SNE%202D).png" width="300" />
    </td>
    <td align="center" valign="top" width="33%">
      <img src="runs/LDM/20260322_211427_053137/plots/MedNIST%20Interpolation.gif" width="300" />
    </td>
  </tr>
  <tr>
    <td align="center" valign="top" width="33%">
      <sub>Diffusion Model - Adversarial Training Curves</sub>
    </td>
    <td align="center" valign="top" width="33%">
      <sub>Diffusion Model - Latent Space (t-SNE 2D)</sub>
    </td>
    <td align="center" valign="top" width="33%">
      <sub>MedNIST Interpolation.gif</sub>
    </td>
  </tr>
</table>

<table cellspacing="18" cellpadding="6">
  <tr>
    <td align="center" valign="top" colspan="3">
      <img src="runs/LDM/20260322_211427_053137/plots/Diffusion%20Model%20-%20Decoded%20Intermediates%20Every%20100%20Steps.png" width="972" />
    </td>
  </tr>
  <tr>
    <td align="center" valign="top" colspan="3">
      <sub>Diffusion Model - Decoded Intermediates Every 100 Steps</sub>
    </td>
  </tr>
</table>

### Run `20260322_225037_317153`

| Metric                             | Value        |
| ---------------------------------- | ------------ |
| Status                             | completed    |
| Batch Size                         | 32           |
| Train/Valid Ratio                  | 0.95         |
| Autoencoder Epochs                 | 60           |
| Diffusion Epochs                   | 80           |
| AE Channels                        | [8, 16, 32]  |
| Latent Channels                    | 24           |
| Residual Blocks                    | 3            |
| Discriminator Layers               | 3            |
| Discriminator Channels             | 16           |
| Diffusion Channels                 | [16, 24, 32] |
| Diffusion Head Channels            | [0, 8, 16]   |
| Diffusion Timesteps                | 1000         |
| Diffusion Beta Start               | 0.0015       |
| Diffusion Beta End                 | 0.0195       |
| Last Validation Loss (Autoencoder) | 0.0527734    |
| Last Validation Loss (Diffusion)   | 0.491641     |
| Run Duration (s)                   | 5042.61      |

<table cellspacing="18" cellpadding="6">
  <tr>
    <td align="center" valign="top" width="33%">
      <img src="runs/LDM/20260322_225037_317153/plots/Diffusion%20Model%20-%20Adversarial%20Training%20Curves.png" width="300" />
    </td>
    <td align="center" valign="top" width="33%">
      <img src="runs/LDM/20260322_225037_317153/plots/Diffusion%20Model%20-%20Latent%20Space%20(t-SNE%202D).png" width="300" />
    </td>
    <td align="center" valign="top" width="33%">
      <img src="runs/LDM/20260322_225037_317153/plots/MedNIST%20Interpolation.gif" width="300" />
    </td>
  </tr>
  <tr>
    <td align="center" valign="top" width="33%">
      <sub>Diffusion Model - Adversarial Training Curves</sub>
    </td>
    <td align="center" valign="top" width="33%">
      <sub>Diffusion Model - Latent Space (t-SNE 2D)</sub>
    </td>
    <td align="center" valign="top" width="33%">
      <sub>MedNIST Interpolation.gif</sub>
    </td>
  </tr>
</table>

<table cellspacing="18" cellpadding="6">
  <tr>
    <td align="center" valign="top" colspan="3">
      <img src="runs/LDM/20260322_225037_317153/plots/Diffusion%20Model%20-%20Decoded%20Intermediates%20Every%20100%20Steps.png" width="972" />
    </td>
  </tr>
  <tr>
    <td align="center" valign="top" colspan="3">
      <sub>Diffusion Model - Decoded Intermediates Every 100 Steps</sub>
    </td>
  </tr>
</table>

### Run `20260323_005636_016702`

| Metric                             | Value           |
| ---------------------------------- | --------------- |
| Status                             | completed       |
| Batch Size                         | 64              |
| Train/Valid Ratio                  | 0.95            |
| Autoencoder Epochs                 | 60              |
| Diffusion Epochs                   | 80              |
| AE Channels                        | [24, 48, 64]    |
| Latent Channels                    | 12              |
| Residual Blocks                    | 3               |
| Discriminator Layers               | 3               |
| Discriminator Channels             | 16              |
| Diffusion Channels                 | [128, 256, 512] |
| Diffusion Head Channels            | [0, 256, 512]   |
| Diffusion Timesteps                | 1000            |
| Diffusion Beta Start               | 0.0015          |
| Diffusion Beta End                 | 0.0195          |
| Last Validation Loss (Autoencoder) | 0.0194259       |
| Last Validation Loss (Diffusion)   | 0.175704        |
| Run Duration (s)                   | 5191.15         |

<table cellspacing="18" cellpadding="6">
  <tr>
    <td align="center" valign="top" width="33%">
      <img src="runs/LDM/20260323_005636_016702/plots/Diffusion%20Model%20-%20Adversarial%20Training%20Curves.png" width="300" />
    </td>
    <td align="center" valign="top" width="33%">
      <img src="runs/LDM/20260323_005636_016702/plots/Diffusion%20Model%20-%20Latent%20Space%20(t-SNE%202D).png" width="300" />
    </td>
    <td align="center" valign="top" width="33%">
      <img src="runs/LDM/20260323_005636_016702/plots/MedNIST%20Interpolation.gif" width="300" />
    </td>
  </tr>
  <tr>
    <td align="center" valign="top" width="33%">
      <sub>Diffusion Model - Adversarial Training Curves</sub>
    </td>
    <td align="center" valign="top" width="33%">
      <sub>Diffusion Model - Latent Space (t-SNE 2D)</sub>
    </td>
    <td align="center" valign="top" width="33%">
      <sub>MedNIST Interpolation.gif</sub>
    </td>
  </tr>
</table>

<table cellspacing="18" cellpadding="6">
  <tr>
    <td align="center" valign="top" colspan="3">
      <img src="runs/LDM/20260323_005636_016702/plots/Diffusion%20Model%20-%20Decoded%20Intermediates%20Every%20100%20Steps.png" width="972" />
    </td>
  </tr>
  <tr>
    <td align="center" valign="top" colspan="3">
      <sub>Diffusion Model - Decoded Intermediates Every 100 Steps</sub>
    </td>
  </tr>
</table>

### Run `20260323_105355_312052`

| Metric                             | Value         |
| ---------------------------------- | ------------- |
| Status                             | completed     |
| Batch Size                         | 64            |
| Train/Valid Ratio                  | 0.95          |
| Autoencoder Epochs                 | 60            |
| Diffusion Epochs                   | 80            |
| AE Channels                        | [16, 24, 32]  |
| Latent Channels                    | 24            |
| Residual Blocks                    | 3             |
| Discriminator Layers               | 3             |
| Discriminator Channels             | 16            |
| Diffusion Channels                 | [64, 64, 128] |
| Diffusion Head Channels            | [0, 64, 128]  |
| Diffusion Timesteps                | 1000          |
| Diffusion Beta Start               | 0.0015        |
| Diffusion Beta End                 | 0.0195        |
| Last Validation Loss (Autoencoder) | 0.0282452     |
| Last Validation Loss (Diffusion)   | 0.215452      |
| Run Duration (s)                   | 3433.38       |

<table cellspacing="18" cellpadding="6">
  <tr>
    <td align="center" valign="top" width="33%">
      <img src="runs/LDM/20260323_105355_312052/plots/Diffusion%20Model%20-%20Adversarial%20Training%20Curves.png" width="300" />
    </td>
    <td align="center" valign="top" width="33%">
      <img src="runs/LDM/20260323_105355_312052/plots/Diffusion%20Model%20-%20Latent%20Space%20(t-SNE%202D).png" width="300" />
    </td>
    <td align="center" valign="top" width="33%">
      <img src="runs/LDM/20260323_105355_312052/plots/MedNIST%20Interpolation.gif" width="300" />
    </td>
  </tr>
  <tr>
    <td align="center" valign="top" width="33%">
      <sub>Diffusion Model - Adversarial Training Curves</sub>
    </td>
    <td align="center" valign="top" width="33%">
      <sub>Diffusion Model - Latent Space (t-SNE 2D)</sub>
    </td>
    <td align="center" valign="top" width="33%">
      <sub>MedNIST Interpolation.gif</sub>
    </td>
  </tr>
</table>

<table cellspacing="18" cellpadding="6">
  <tr>
    <td align="center" valign="top" colspan="3">
      <img src="runs/LDM/20260323_105355_312052/plots/Diffusion%20Model%20-%20Decoded%20Intermediates%20Every%20100%20Steps.png" width="972" />
    </td>
  </tr>
  <tr>
    <td align="center" valign="top" colspan="3">
      <sub>Diffusion Model - Decoded Intermediates Every 100 Steps</sub>
    </td>
  </tr>
</table>

### Run `20260323_120641_328800`

| Metric | Value |
|---|---|
| Status | completed |
| Batch Size | 64 |
| Train/Valid Ratio | 0.95 |
| Autoencoder Epochs | 60 |
| Diffusion Epochs | 80 |
| AE Channels | [24, 48, 64] |
| Latent Channels | 12 |
| Residual Blocks | 3 |
| Discriminator Layers | 3 |
| Discriminator Channels | 16 |
| Diffusion Channels | [128, 256, 512] |
| Diffusion Head Channels | [0, 256, 512] |
| Diffusion Timesteps | 1000 |
| Diffusion Beta Start | 0.0015 |
| Diffusion Beta End | 0.0195 |
| Last Validation Loss (Autoencoder) | 0.0299948 |
| Last Validation Loss (Diffusion) | 0.227937 |
| Run Duration (s) | 3938.95 |

<table cellspacing="18" cellpadding="6">
  <tr>
    <td align="center" valign="top" width="33%">
      <img src="runs/LDM/20260323_120641_328800/plots/Diffusion%20Model%20-%20Adversarial%20Training%20Curves.png" width="300" />
    </td>
    <td align="center" valign="top" width="33%">
      <img src="runs/LDM/20260323_120641_328800/plots/Diffusion%20Model%20-%20Latent%20Space%20(t-SNE%202D).png" width="300" />
    </td>
    <td align="center" valign="top" width="33%">
      <img src="runs/LDM/20260323_120641_328800/plots/MedNIST%20Interpolation.gif" width="300" />
    </td>
  </tr>
  <tr>
    <td align="center" valign="top" width="33%">
      <sub>Diffusion Model - Adversarial Training Curves</sub>
    </td>
    <td align="center" valign="top" width="33%">
      <sub>Diffusion Model - Latent Space (t-SNE 2D)</sub>
    </td>
    <td align="center" valign="top" width="33%">
      <sub>MedNIST Interpolation.gif</sub>
    </td>
  </tr>
</table>

<table cellspacing="18" cellpadding="6">
  <tr>
    <td align="center" valign="top" colspan="3">
      <img src="runs/LDM/20260323_120641_328800/plots/Diffusion%20Model%20-%20Decoded%20Intermediates%20Every%20100%20Steps.png" width="972" />
    </td>
  </tr>
  <tr>
    <td align="center" valign="top" colspan="3">
      <sub>Diffusion Model - Decoded Intermediates Every 100 Steps</sub>
    </td>
  </tr>
</table>

## Observations

I have tried and compared 2 scenarios: the VAE in an adversarial setting, and the VAE with a Diffusion model in that same setting.

I made a point to make both runs as comparable as possible, because I wanted to see what the use of a diffusion model brought to the table.

In an attempt at improving the results, I have tried to:

- Increase the latent space dimension from 3 to 12 to 24
- Increase the diffusion model's channels from 128, 128, 256 to 128, 256, 512
- Increase the number of residual blocks from 2 to 3
- Increase the number of channels of the variational auto encoder from (16, 32, 64) to (24, 48, 64)
- Increase the adversarial weight from 1e-2 to 1e-2

The change that proved to be most significant was the increase of the diffusion model's channels. While the VAE's important, its main goal is only to compress the images into meaningful features. The diffusion model is the one which actually works hard generating convincing images.

Case in point: when I tried to drastically decrease the diffusion model channels to [16, 24, 32], the decoded images every 100 steps did have the time to converge to a hand. While the latent space was convincingly distributed, the gif was visibly blurry, similarly to the GAN.

**When it came to the results of the VAE-only GAN (all):**

Interpolated gif showed visible hands, albeit blurry. Intermediate visualisations, as it went through the latent space, looked less convincing and more scary, but overall a better result that originally saw in class (which had a latent space of 3).

**When it came to the results of the Latent Diffusion Model (20260323_005636_016702):**

While the adversarial training curves look a bit odd in the beginning, they eventually stabilise to a result that is expected, and the visual result, be it in Diffusion Model - Latent Space (t-SNE 2D).png, MedNIST Interpolation.gif or Diffusion Model - Decoded Intermediates Every 100 Steps.png look beautiful, and what I would expect from a good model. The hands are relatively crisp and convincing, and the decoded intermediates looked normal for once.

For the rest of the results, I tried significantly lower diffusion channels, so these runs showed lower quality results.

I think it is also worth noting that the LDM is significantly slower than the GAN. It takes roughly an hour and a half to run on my RTX 4060Ti with 16 GB of VRAM, compared to 20 minutes give or take a minute of train time for the VAE-only GAN.

I have also tried to only increase latent space (GAN: 20260323_103245_605233, LDM: 20260323_105355_312052) to 24, and from what I observed, results were rather poor on both sides, and more noticeably so on the LDM's Diffusion Model - Decoded Intermediates Every 100 Steps.png file where a full hand could not be reconstructed even after the last steps.

All in all, my best result, in my opinion, was with the Latent Diffusion Model with the Diffusion Model having [128, 256, 512] channels with 12 latent channels.
