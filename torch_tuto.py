import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

# Vérifier si un GPU est disponible et utiliser CUDA si possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation de : {device}")

# Transformations des images (mise en échelle et normalisation)
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=3),  # Convertir en 3 canaux (RVB)
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# Charger le dataset CIFAR-10
trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=64, shuffle=False, num_workers=2
)

# Classes des images
classes = (
    "avion",
    "automobile",
    "oiseau",
    "chat",
    "cerf",
    "chien",
    "grenouille",
    "cheval",
    "bateau",
    "camion",
    "maison",
    "chalet",
    "pont",
    "route",
    "personne",
    "panneau de signalisation",
    "train",
    "arbre",
    "feu"
)

# Définir le nombre de classes
num_classes = 10  # Exemple pour MNIST, adapte selon ton dataset


# Définition du réseau CNN amélioré avec Dropout
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # Convolutions
        # Modifier la première couche de convolution pour accepter 3 canaux (images RVB)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 3 canaux pour RVB
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Couches entièrement connectées avec Dropout
        self.fc1 = nn.Linear(
            64 * 8 * 8, 1024
        )  # 64*8*8 = 4096, correspond au tensor aplati
        self.dropout1 = nn.Dropout(0.5)  # Dropout à 50%
        self.fc2 = nn.Linear(1024, 512)  # 1024 neurones en entrée, 512 en sortie
        self.dropout2 = nn.Dropout(0.5)  # Dropout à 50%
        self.fc3 = nn.Linear(512, num_classes)  # 10 classes pour CIFAR-10

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Convolution + ReLU + MaxPooling
        x = self.pool(
            F.relu(self.conv2(x))
        )  # Nouvelle couche de convolution + ReLU + MaxPooling
        # print(x.shape)  # Affiche la taille du tensor pour vérifier
        x = x.view(
            x.size(0), -1
        )  # Aplatir automatiquement en fonction de la taille du batch
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # Appliquer Dropout après fc1
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)  # Appliquer Dropout après fc2
        x = self.fc3(x)
        return x


def train_model(model, trainloader, criterion, optimizer, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(
                device
            )  # Envoi des données sur GPU

            optimizer.zero_grad()  # Réinitialisation des gradients
            outputs = model(images)  # Passage avant (forward)
            loss = criterion(outputs, labels)  # Calcul de la perte
            loss.backward()  # Rétropropagation
            optimizer.step()  # Mise à jour des poids

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(trainloader):.4f}")

    print("Entraînement terminé !")


def save_model(
    model,
    optimizer,
    epoch,
    loss,
    model_path="model_cifar10_glm.pth",
    checkpoint_path="checkpoint.pth",
):
    torch.save(model.state_dict(), model_path)  # Sauvegarde du modèle
    print(f"Modèle sauvegardé sous {model_path}")

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        checkpoint_path,
    )
    print(f"Checkpoint sauvegardé sous {checkpoint_path}")


def load_model(model, optimizer, checkpoint_path="checkpoint.pth"):
    checkpoint = torch.load(checkpoint_path)  # Charger le checkpoint
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    model.train()  # Remettre en mode entraînement
    print(f"Reprise de l'entraînement à l'époque {epoch}")
    return epoch, loss


def evaluate_model(model, testloader):
    correct = 0
    total = 0
    model.eval()  # Mode évaluation (désactive dropout et batch norm)
    with torch.no_grad():  # Pas de calcul de gradient
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Précision du modèle : {accuracy:.2f}%")


def predict_image(model, image_path):
    model.eval()  # Mode évaluation
    transform = transforms.Compose(
        [
            transforms.Resize(
                (32, 32)
            ),  # Redimensionner l'image à la taille attendue par le modèle
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    image = Image.open(image_path)
    image = (
        transform(image).unsqueeze(0).to(device)
    )  # Ajouter une dimension pour le batch et envoyer sur GPU

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return classes[predicted.item()]


def main():
    model = CNNModel().to(device)
    criterion = nn.CrossEntropyLoss()  # Fonction de perte pour classification
    optimizer = optim.Adam(
        model.parameters(), lr=0.001, weight_decay=0.0001
    )  # Optimiseur Adam avec weight_decay

    # Entraînement du modèle
    print("Entraînement du modèle...")
    # train_model(model, trainloader, criterion, optimizer, num_epochs=40)

    # Sauvegarde du modèle et du checkpoint
    print("Sauvegarde du modèle...")
    # save_model(
    #    model, optimizer, epoch=40, loss=0.0
    # )  # Remplacer 0.0 par la perte finale

    # Évaluation du modèle
    print("Évaluation du modèle...")
    evaluate_model(model, testloader)

    # Charger le modèle et reprendre l'entraînement si nécessaire
    print("Chargement du modèle...")
    epoch, loss = load_model(model, optimizer)

    # Exemple d'utilisation
    print("Prédiction d'une image...")
    image_path = "minou.jpg"
    predicted_class = predict_image(model, image_path)
    print(f"L'image est prédite comme étant : {predicted_class}")


if __name__ == "__main__":
    main()
