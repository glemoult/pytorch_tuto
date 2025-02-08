import torch

# Création d'un tenseur sur le GPU
x = torch.randn(3, 3).to("cuda")
print("Tenseur sur GPU :", x)

# Vérifier si le tenseur est bien sur le GPU
print("Le tenseur est sur :", x.device)
