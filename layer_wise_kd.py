import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CyclicLR, CosineAnnealingLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score
from utils.eval_utils import *
from utils.visualize_utils import *
from utils.prune_utils import *
from utils.train_utils import *
from models.Resnet import *
from models.VGG import *
from copy import deepcopy
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device:", device)

torch.manual_seed(42)

train_transform = transforms.Compose(
    [
        ImgAugTransform_Large(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
        ),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
        ),
    ]
)

full_dataset = datasets.CIFAR10(
    root="./data/Cifar-10", train=True, download=False, transform=train_transform
)
test_dataset = datasets.CIFAR10(
    root="./data/Cifar-10", train=False, download=False, transform=test_transform
)

train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

val_dataset.dataset.transform = test_transform

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"Number of Train samples: {len(train_dataset)}")
print(f"Number of Val samples: {len(val_dataset)}")
print(f"Number of Test samples: {len(test_dataset)}")


def iterative_filter_pruning(
    scheduler="cosine",
    prune_steps=1,
    prune_ratio=[0.2],
    epochs=20,
    lr=1e-3,
    optim="adam",
    exp_name="ToyExp_Global_Iter_Adaptivity_Resnet20",
    hints=True,
):
    assert prune_steps == len(prune_ratio), (
        f"Expected Prune steps ({prune_steps}) to match the length of prune ratios ({len(prune_ratio)})."
    )

    if hints:
        original_model = Resnet_General(channel_dict=None, KD=True, pre_train=True).to(device)
        original_model.load_state_dict(torch.load("./pre-trained/resnet18_pre-trained_cifar10.pth"), strict=False)
    else:
        original_model = torch.hub.load(
            "chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True
        ).to(device)

    accuracy_list = []
    profiling = {"params": [], "MAC's": [], "model_size": []}

    print(f"{'-' * 20}Starting with Experiment {exp_name}{'-' * 20}")
    acc_original, f1_original, params_original, mac_original, size_mb_original = (
        evaluate(original_model, test_loader, profiling=True)
    )
    accuracy_list.append(acc_original)
    profiling["params"].append(params_original)
    profiling["MAC's"].append(mac_original)
    profiling["model_size"].append(size_mb_original)

    orig_copy = deepcopy(original_model)
    unprune_meta_data = []

    # Iterative Pruning of Original Model
    for i in range(prune_steps):
        original_model, pruned_index = Index_Aware_pruner(
            original_model, amount=prune_ratio[i], prune_step=i, plot=False, BN=False
        )
        non_zero_channels_info, num_unpruned_channels, unpruned_weights = (
            get_unpruned_indices_and_counts(original_model)
        )
        unprune_meta_data.append(
            {
                "non_zero_channels_info": non_zero_channels_info,
                "num_unpruned_channels": num_unpruned_channels,
                "unpruned_weights": unpruned_weights,
            }
        )

    # with open(f"./saves/pruning_metadata/{exp_name}_unprune_data.txt", "w") as file:
    #     for index, dictionary in enumerate(unprune_meta_data):
    #         file.write(f"Entry {index + 1}:\n")
    #         for key, value in dictionary.items():
    #             file.write(f"  {key}: {value}\n")
    #         file.write("\n")  # Add a blank line between entries

    # Core Model Training
    if hints:
        core_model = Resnet_General(unprune_meta_data[-1]["num_unpruned_channels"], KD=True).to(
            device
        )
    else:
        core_model = Resnet_General(unprune_meta_data[-1]["num_unpruned_channels"]).to(
            device
        )
    copy_weights_from_dict(
        core_model, unprune_meta_data[-1]["unpruned_weights"], None, False
    )
    # fine_tuner(
    #     core_model,
    #     train_loader,
    #     val_loader,
    #     num_epochs=epochs,
    #     optim_type=optim,
    #     scheduler_type=scheduler,
    #     exp_name=f"{exp_name}_Training_core_model_Level-{str(prune_steps + 1)}",
    #     LR=lr,
    # )

    # KD_trainer_zerograd(
    #     teacher=orig_copy,
    #     student=core_model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     lr=lr,
    #     alpha=0.75,
    #     beta=3,
    #     freeze_dim0=None,
    #     freeze_dim1=None,
    #     num_epochs=epochs,
    #     exp_name=f"{exp_name}_Training_model_Level-{i + 1}",
    #     zero_out_grad=False,
    # )

    KD_trainer(
        student=core_model,
        teacher=orig_copy,
        train_loader=train_loader,
        val_loader=val_loader,
        teacher_channels=64,
        student_channels=unprune_meta_data[-1]["num_unpruned_channels"]['layer3.2.conv2'][0],
        lr=lr,
        alpha=0.75,
        beta=3,
        num_epochs=epochs,
        exp_name=f"{exp_name}_Training_model_Level-{i + 1}",
        KD_variant="hints")

    acc_pruned, f1_pruned, params_core, mac_core, size_mb_core = evaluate(
        core_model, test_loader, profiling=True
    )

    save_statedict(
        core_model,
        unprune_meta_data[-1]["num_unpruned_channels"],
        path=f"./saves/state_dicts/{exp_name}_ResNet18_Core_Model.pth",
    )
    accuracy_list.append(acc_pruned)
    profiling["params"].append(params_core)
    profiling["MAC's"].append(mac_core)
    profiling["model_size"].append(size_mb_core)

    return accuracy_list, profiling


if __name__ == "__main__":
    lr = 1e-3
    epochs = 10

    results = []
    trial = 0
    EXP = "layer_wise_kd"
    sch = "cosine"

    test_acc, profiling = iterative_filter_pruning(
        scheduler=sch,
        epochs=epochs,
        lr=lr,
        exp_name=f"Exp_{trial + 1}_{EXP}_ResNet18_{sch.upper()}_{str(lr)}",
    )

    results.append(
        {
            "Trial": trial + 1,
            "Learning Rate": lr,
            "Epochs": epochs,
            "Scheduler": sch.upper(),
            "Test Accuracy (Original)": test_acc[0],
            "Test Accuracy (Core Model)": test_acc[1],
            "Test Accuracy (Descendant Models)": test_acc[2:],
            "Num_Params": profiling["params"],
            "MAC's": profiling["MAC's"],
            "Model Size (MB)": profiling["model_size"],
        }
    )

    print(f"{'-' * 20}EXPERIMENT FINISHED{'-' * 20}")
    exit()
