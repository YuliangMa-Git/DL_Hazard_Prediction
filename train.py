import time
import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from network import HazardPredictionNN, HazardPredictionImageNN
from utils import loss_fn, model_evaluation
from dataset_upload import DatasetProcessor
from test_dataset_upload import TestDatasetProcessor


def main(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_processor = DatasetProcessor(
        args.train_path,
        args.train_base_waypoint_path
    )
    camera_train, joints_train, waypoints_train, targets_train, rgb_images_train, depth_images_train = train_processor.process_dataset()

    train_set = TensorDataset(
        torch.FloatTensor(camera_train),
        torch.FloatTensor(joints_train),
        torch.FloatTensor(waypoints_train),
        torch.FloatTensor(targets_train),
        torch.FloatTensor(rgb_images_train),
        torch.FloatTensor(depth_images_train)
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    test_processor = TestDatasetProcessor(
        args.test_path,
        args.test_base_waypoint_path
    )
    camera_test, joints_test, waypoints_test, targets_test, rgb_images_test, depth_images_test = test_processor.process_dataset()

    test_set = TensorDataset(
        torch.FloatTensor(camera_test),
        torch.FloatTensor(joints_test),
        torch.FloatTensor(waypoints_test),
        torch.FloatTensor(targets_test),
        torch.FloatTensor(rgb_images_test),
        torch.FloatTensor(depth_images_test)

    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # model = HazardPredictionNN(num_classes=1).to(device) # Choose the model
    model = HazardPredictionImageNN(num_classes=1).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)

    best_loss = float('inf')
    train_loss_history = []
    test_loss_history = []

    print("\n---------- Starting training ----------")
    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0

        for batch_idx, (camera, joints, waypoints, labels, rgb_images, depth_images) in enumerate(train_loader):

            B = camera.shape[0]
            camera = camera.unsqueeze(1)
            camera = camera.reshape(B, 25, 3).to(device)  # human pose coordinates in camera frame
            joints = joints.to(device)
            waypoints = waypoints.unsqueeze(1)
            waypoints = waypoints.reshape(B, 64, 7).to(device)
            rgb_images = rgb_images.to(device)
            depth_images = depth_images.to(device)
            labels = labels.to(device)

            # outputs = model(camera, joints, waypoints) # Up to the model
            outputs = model(rgb_images, depth_images, joints, waypoints)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{args.epochs}] Batch [{batch_idx + 1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        train_loss_history.append(epoch_loss)

        test_loss = model_evaluation(test_loader, model, device)
        test_loss_history.append(test_loss)

        scheduler.step(test_loss)

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), f"best_model_epoch.pth")
            print(f"Model got the best training at Epoch {epoch + 1}, Training loss: {test_loss:.4f}")

        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch + 1}/{args.epochs} Summarize: "
              f"Training loss = {epoch_loss:.4f}, Test loss = {test_loss:.4f}, "
              f"Time = {epoch_time:.1f}s, "
              f"Learning rate = {optimizer.param_groups[0]['lr']:.2e}\n" + "-" * 50)

    plt.figure(figsize=(12, 6))
    plt.plot(train_loss_history, label='Training loss', color='blue', linewidth=2)
    plt.plot(test_loss_history, label='Test loss', color='red', linestyle='--', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training loss and test loss', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nTraining finished!")
    print(f"Best test loss: {best_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hazard prediction')

    parser.add_argument("--train_path", type=str, default='train_dataset',
                        help="path of train dataset")
    parser.add_argument("--test_path", type=str, default='test_dataset',
                        help="path of test dataset")
    parser.add_argument("--train_base_waypoint_path", type=str, default='train_dataset',
                        help="path of train dataset")
    parser.add_argument("--test_base_waypoint_path", type=str, default='test_dataset',
                        help="path of test dataset")

    parser.add_argument("--seed", type=int, default=230,
                        help="seed (default: 42)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="epoch (default: 100)")
    parser.add_argument("--train_batch_size", type=int, default=16,
                        help="train batch (default: 32)")
    parser.add_argument("--test_batch_size", type=int, default=64,
                        help="test batch (default: 64)")
    parser.add_argument("--learning_rate", type=float, default=0.0005,
                        help="Lr (default: 0.001)")
    parser.add_argument("--weight_decay", type=float, default=0.00015,
                        help="weight decay (default: 0.0001)")

    args = parser.parse_args()

    assert os.path.exists(args.train_path), f"path does not exist: {args.train_path}"
    assert os.path.exists(args.test_path), f"path does not exist: {args.test_path}"

    main(args)
