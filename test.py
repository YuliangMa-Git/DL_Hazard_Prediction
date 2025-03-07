import torch
import argparse
from torch.utils.data import DataLoader, TensorDataset
import os
from network import HazardPredictionNN, HazardPredictionImageNN
from test_dataset_upload import TestDatasetProcessor
from utils import model_evaluation, get_F1_measure


def test_all(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on: {device}")

    # model = HazardPredictionNN(num_classes=1).to(device)
    model = HazardPredictionImageNN(num_classes=1).to(device)

    model_path = './best_model_epoch.pth'

    if not torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    else:
        model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Successfully load model: {model_path}")

    processor = TestDatasetProcessor(
        args.test_path,
        args.test_base_waypoint_path
    )
    camera, joints, waypoints, targets, rgb_images, depth_images = processor.process_dataset()

    assert waypoints.shape[1] == 448, f"Waypoints dimension should be 448，actual dimension is {waypoints.shape[1]}"

    test_set = TensorDataset(
        torch.FloatTensor(camera),
        torch.FloatTensor(joints),
        torch.FloatTensor(waypoints),
        torch.FloatTensor(targets),
        torch.FloatTensor(rgb_images),
        torch.FloatTensor(depth_images)
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print("\n---------- Start testing ----------")

    test_loss = model_evaluation(test_loader, model, device)
    print(f"Test loss: {test_loss:.4f}")

    get_F1_measure(test_loader, model, device, 0.5)

    print("\nTest finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hazard prediction_test')

    parser.add_argument("--test_batch_size", type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument("--test_path", type=str, default='test_dataset',
                        help="Path of test dataset")
    parser.add_argument("--test_base_waypoint_path", type=str, default='test_dataset',
                        help="Path of test dataset")

    args = parser.parse_args()

    assert os.path.exists(args.test_path), f"path does not exist: {args.test_path}"

    test_all(args)
