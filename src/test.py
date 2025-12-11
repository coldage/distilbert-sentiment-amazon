# cuda 13.0, CUDA 12.8 PyTorch
# pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130import torch

import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("CUDA Version:", torch.version.cuda)
    print("GPU Memory:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")
    print("\n✓ GPU is ready to use!")
else:
    print("\n✗ GPU is not available")

#update treiber über nvidia app
# get new ver num: (in normalem einfach) nvidia-smi
# get new link to install pytorch (website link in cursor oder in notes datei)
# pytorch installation in NLP env in conda!, pytorch ist jetzt komplett deinstalliert
# hoffentlich updated sich cuds dass ich da eine besser version finden kann
# in der annahme dass das alles funktioniert hat wie gaplant, die test datei versuchen zu machen
# wenn test datei gpu okay anzeigt, dann das ganze über nacht einmal laufen lassen