import torch
import torch.nn as nn
from train_utils import TransformationLayer, HintLoss
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_student_layer_wise(student, teacher, train_loader, val_loader, teacher_channels, student_channels, lr=0.001, alpha=0.75, device=device):
    teacher = teacher.to(device).eval()
    student = student.to(device).train()

    transformation_layers = []

    # Initialize the transformation layers
    for (student_channel, teacher_channel) in list(zip(student_channel, teacher_channel)):
        transformation_layer.append()
        transformation_layer = TransformationLayer(student_channels, teacher_channels).to(device)
        hint_loss_fn = HintLoss()
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(list(student.parameters()) + list(transformation_layer.parameters()), lr=lr)

    train_total_loss = 0.0
    train_total_correct = 0
    train_total_samples = 0
    train_total_hint_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        student_outputs, inter_student = student(inputs)
        with torch.no_grad():
            _, inter_teacher = teacher(inputs)

        # Compute losses
        classification_loss = criterion(student_outputs, labels)
        hint_loss = hint_loss_fn(transformation_layer(inter_student), inter_teacher)

        loss = alpha * classification_loss + (1 - alpha) * hint_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update train metrics
        train_total_loss += loss.item() * inputs.size(0)
        train_total_hint_loss += hint_loss.item() * inputs.size(0)
        _, predicted = torch.max(student_outputs, dim=1)
        train_total_correct += (predicted == labels).sum().item()
        train_total_samples += inputs.size(0)

    train_avg_loss = train_total_loss / train_total_samples
    train_avg_hint_loss = train_total_hint_loss / train_total_samples
    train_accuracy = train_total_correct / train_total_samples

    # Validation looph
    student.eval()

    val_total_loss = 0.0
    val_total_correct = 0
    val_total_samples = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs, _ = student(inputs)
            loss = criterion(outputs, labels)

            # Update validation metrics
            val_total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, dim=1)
            val_total_correct += (predicted == labels).sum().item()
            val_total_samples += inputs.size(0)

    val_avg_loss = val_total_loss / val_total_samples
    val_accuracy = val_total_correct / val_total_samples

    return train_avg_loss, train_avg_hint_loss, train_accuracy, val_avg_loss, val_accuracy

