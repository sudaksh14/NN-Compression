import torch
import torch.nn as nn
from utils.train_utils import TransformationLayer, HintLoss
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_student_layer_wise(
    student,
    teacher,
    train_loader,
    val_loader,
    teacher_channels,
    student_channels,
    lr=0.001,
    alpha=0.75,
    device=device,
):
    teacher = teacher.to(device).eval()
    student = student.to(device).train()

    transformation_layers = []

    # Initialize the transformation layers
    for student_channel, teacher_channel in list(zip(student_channel, teacher_channel)):
        transformation_layer.append()
        transformation_layer = TransformationLayer(
            student_channels, teacher_channels
        ).to(device)
        hint_loss_fn = HintLoss()
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        list(student.parameters()) + list(transformation_layer.parameters()), lr=lr
    )

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

    return (
        train_avg_loss,
        train_avg_hint_loss,
        train_accuracy,
        val_avg_loss,
        val_accuracy,
    )


def KD_LW_trainer(
    student,
    teacher,
    train_loader,
    val_loader,
    lr,
    alpha,
    beta,
    omega,
    student_channels,
    teacher_channels,
    num_epochs=20,
    exp_name="KD LW",
    KD_variant="layer_wise",
):
    teacher = teacher.to(device)
    student = student.to(device)

    if KD_variant == "layer_wise":
        transformation_layer1 = TransformationLayer(
            student_channels[0], teacher_channels[0]
        ).to(device)
        transformation_layer2 = TransformationLayer(
            student_channels[1], teacher_channels[1]
        ).to(device)
        transformation_layer3 = TransformationLayer(
            student_channels[2], teacher_channels[2]
        ).to(device)
        hint_loss_fn = HintLoss()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            list(student.parameters())
            + list(transformation_layer1.parameters())
            + list(transformation_layer2.parameters())
            + list(transformation_layer3.parameters()),
            lr=lr,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
    elif KD_variant == "hints":
        transformation_layer = TransformationLayer(
            student_channels, teacher_channels
        ).to(device)
        hint_loss_fn = HintLoss()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            list(student.parameters()) + list(transformation_layer.parameters()), lr=lr
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(student.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

    if KD_variant == "hints":
        train_hint_losses = []
    elif KD_variant == "layer-wise":
        train_layer_wise_losses = []
    else:
        train_distillation_losses = []
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        train_total_loss = 0.0
        train_total_correct = 0
        train_total_samples = 0
        train_total_distillation_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            student_outputs, inter_student1, inter_student2, inter_student3 = student(
                inputs
            )
            with torch.no_grad():
                teacher_outputs, inter_teacher1, inter_teacher2, inter_teacher3 = (
                    teacher(inputs)
                )

            # Compute losses
            classification_loss = criterion(student_outputs, labels)
            distillation_loss = F.kl_div(
                F.log_softmax(student_outputs / beta, dim=1),
                F.softmax(teacher_outputs / beta, dim=1),
                reduction="batchmean",
            ) * (beta**2)
            hint_loss = (
                hint_loss_fn(transformation_layer1(inter_student1), inter_teacher1)
                + hint_loss_fn(transformation_layer2(inter_student2), inter_teacher2)
                + hint_loss_fn(transformation_layer3(inter_student3), inter_teacher3)
            )

            # Experimentation will need to be done to find the optimal alpha and omega
            loss = alpha * classification_loss + (1 - alpha) * (
                omega * distillation_loss + (1 - omega) * hint_loss
            )

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update train metrics
            train_total_loss += loss.item() * inputs.size(0)
            train_total_distillation_loss += distillation_loss.item() * inputs.size(0)
            _, predicted = torch.max(student_outputs, dim=1)
            train_total_correct += (predicted == labels).sum().item()
            train_total_samples += inputs.size(0)

        train_avg_loss = train_total_loss / train_total_samples
        train_avg_distillation_loss = (
            train_total_distillation_loss / train_total_samples
        )
        train_accuracy = train_total_correct / train_total_samples

        # Validation loop
        student.eval()

        val_total_loss = 0.0
        val_total_correct = 0
        val_total_samples = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs, _, _, _ = student(inputs)
                loss = criterion(outputs, labels)

                # Update validation metrics
                val_total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, dim=1)
                val_total_correct += (predicted == labels).sum().item()
                val_total_samples += inputs.size(0)

        val_avg_loss = val_total_loss / val_total_samples
        val_accuracy = val_total_correct / val_total_samples

        train_losses.append(train_avg_loss)
        train_distillation_losses.append(train_avg_distillation_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_avg_loss)
        val_accuracies.append(val_accuracy)

        scheduler.step()

        print(
            f"Epoch [{epoch + 1}/{num_epochs}]: Train Loss: {train_avg_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_avg_loss:.4f}, Val Acc: {val_accuracy:.4f}"
        )

    print("Knowledge Distillation Training Complete")
