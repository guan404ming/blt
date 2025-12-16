# Vendored Code Notice

This directory contains vendored code from the Wav2Lip project.

## Source

- **Project**: Wav2Lip - A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild
- **Repository**: https://github.com/Rudrabha/Wav2Lip
- **Commit**: f1ddb368254737f4531c41fc2976e6dfda1b4924
- **License**: MIT License (see README.md for full license text)

## What's Included

We've vendored only the essential files needed for inference:
- `inference.py` - Main inference script
- `audio.py` - Audio processing utilities
- `hparams.py` - Hyperparameters
- `face_detection/` - Face detection module
- `models/` - Wav2Lip model definitions

## Modifications

- `audio.py:100` - Updated librosa.filters.mel() to use keyword arguments for compatibility with librosa>=0.10

## Why Vendored?

The full Wav2Lip repository is 421MB with many unnecessary files (training scripts, evaluation tools, datasets). By vendoring only essential files (~290KB), we reduce repository size by 99.9% while maintaining full functionality.

## Citation

If you use this code, please cite the original Wav2Lip paper:

```
@inproceedings{prajwal2020lip,
  title={A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild},
  author={Prajwal, K R and Mukhopadhyay, Rudrabha and Namboodiri, Vinay P and Jawahar, C V},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
  year={2020}
}
```
