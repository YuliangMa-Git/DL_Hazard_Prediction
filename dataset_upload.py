import os
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import random


class DatasetProcessor:
    def __init__(self, dataset_dir, base_waypoint_path):
        self.dataset_dir = dataset_dir
        self.base_waypoint_path = os.path.abspath(
            os.path.join(r'path/to/project', base_waypoint_path)
        )
        self.label_mapping = {0: 0, 1: 1}
        self.num_classes = len(self.label_mapping)

        self.image_transform = transforms.Compose([
            transforms.Resize((240, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.depth_transform = transforms.Compose([
            transforms.Resize((240, 320)),
            transforms.ToTensor()
        ])
        self.flip_transform = transforms.RandomHorizontalFlip(p=1)

    def load_waypoints_data(self, waypoint_path):
        try:
            waypoint_data = pd.read_csv(waypoint_path)
            return waypoint_data.values
        except Exception as e:
            print(f"can not load file {waypoint_path}: {e}")
            return np.zeros((1, 301))

    def load_and_process_image(self, image_path, is_depth=False):
        try:
            image = Image.open(image_path).convert('RGB' if not is_depth else 'L')
            image = self.image_transform(image) if not is_depth else self.depth_transform(image)
            return image
        except Exception as e:
            print(f"can not load image {image_path}: {e}")
            return torch.zeros(3, 240, 320) if not is_depth else torch.zeros(1, 240, 320)

    def process_dataset(self):
        camera_values_list, joint_states_list, waypoints_data_list, targets_list = [], [], [], []
        rgb_images_list, depth_images_list = [], []

        for subfolder in os.listdir(self.dataset_dir):
            subfolder_path = os.path.join(self.dataset_dir, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            label_csv_path = os.path.join(subfolder_path, 'label.csv')
            if not os.path.exists(label_csv_path):
                continue

            try:
                df = pd.read_csv(label_csv_path)
                valid_rows = df['Category'].isin(self.label_mapping.keys())

                if valid_rows.sum() == 0:
                    continue

                # Features
                camera_values = pd.concat([
                    df[[f'X_{i}' for i in range(25)]],
                    df[[f'Y_{i}' for i in range(25)]],
                    df[[f'D_{i}' for i in range(25)]]
                ], axis=1).values[valid_rows]

                flattened_camera_values = np.array([
                    np.column_stack([row[:25], row[25:50], row[50:75]]).flatten()
                    for row in camera_values
                ])

                joint_states = df[['panda_joint1', 'panda_joint2', 'panda_joint3',
                                   'panda_joint4', 'panda_joint5', 'panda_joint6',
                                   'panda_joint7']].values[valid_rows]
                filtered_labels = df.loc[valid_rows, 'Category'].map(self.label_mapping).values.astype(np.int64)

                # Waypoints
                waypoints_data = [self.load_waypoints_data(os.path.join(subfolder_path, wp)).flatten()
                                  for wp in df.loc[valid_rows, 'Waypoints']]

                # Images
                for rgb_path, depth_path in zip(df.loc[valid_rows, 'RGB_image'], df.loc[valid_rows, 'Depth_image']):
                    full_rgb_path = os.path.join(subfolder_path, rgb_path)
                    full_depth_path = os.path.join(subfolder_path, depth_path)

                    rgb_images_list.append(self.load_and_process_image(full_rgb_path, is_depth=False))
                    depth_images_list.append(self.load_and_process_image(full_depth_path, is_depth=True))

                # Append processed data
                camera_values_list.append(flattened_camera_values)
                joint_states_list.append(joint_states)
                waypoints_data_list.append(np.array(waypoints_data))
                targets_list.append(filtered_labels)

            except Exception as e:
                print(f"处理 {subfolder} 失败: {e}")

        # Stack data
        camera_values = np.vstack(camera_values_list)
        joint_states = np.vstack(joint_states_list)
        waypoints_data = np.vstack(waypoints_data_list)
        targets = np.concatenate(targets_list)
        rgb_images = torch.stack(rgb_images_list)
        depth_images = torch.stack(depth_images_list)

        print(f"data shape: camera={camera_values.shape}, joints={joint_states.shape}, "
              f"waypoints={waypoints_data.shape}, targets={targets.shape}, "
              f"RGB images={rgb_images.shape}, Depth images={depth_images.shape}")

        # Apply oversampling (duplicate non-zero labels)
        camera_values, joint_states, waypoints_data, targets, rgb_images, depth_images = self.augment_data(
            camera_values, joint_states, waypoints_data, targets, rgb_images, depth_images)

        return camera_values, joint_states, waypoints_data, targets, rgb_images, depth_images

    def augment_data(self, camera, joints, waypoints, targets, rgb_images, depth_images):
        """Applies augmentation with 50% probability to all samples, ensuring consistency across all data types."""
        rgb_images_augmented = []
        depth_images_augmented = []
        waypoints_augmented = []
        camera_augmented = []
        joints_augmented = []
        targets_augmented = []

        for i in range(len(rgb_images)):
            # Always add original data
            rgb_images_augmented.append(rgb_images[i].unsqueeze(0))
            depth_images_augmented.append(depth_images[i].unsqueeze(0))
            waypoints_augmented.append(waypoints[i])
            camera_augmented.append(camera[i])
            joints_augmented.append(joints[i])
            targets_augmented.append(targets[i])

            # Augment hazard samples
            if targets[i] == 1:
                flipped_rgb = self.vertical_flip_transform(rgb_images[i]).unsqueeze(0)
                flipped_depth = self.vertical_flip_transform(depth_images[i]).unsqueeze(0)

                # Flip waypoints, camera, and joints (assuming y-axis flip)
                flipped_waypoints = waypoints[i].copy()
                flipped_waypoints[1] = -flipped_waypoints[1]

                flipped_camera = camera[i].copy()
                flipped_camera[1] = -flipped_camera[1]

                flipped_joints = joints[i].copy()
                flipped_joints[1] = -flipped_joints[1]

                # Append augmented data
                rgb_images_augmented.append(flipped_rgb)
                depth_images_augmented.append(flipped_depth)
                waypoints_augmented.append(flipped_waypoints)
                camera_augmented.append(flipped_camera)  # Fix: Add flipped camera data
                joints_augmented.append(flipped_joints)  # Fix: Add flipped joints data
                targets_augmented.append(targets[i])  # Fix: Duplicate target for the augmented version

        # Convert lists to tensors/arrays
        rgb_images_augmented = torch.cat(rgb_images_augmented, dim=0)
        depth_images_augmented = torch.cat(depth_images_augmented, dim=0)
        waypoints_augmented = np.vstack(waypoints_augmented)
        camera_augmented = np.vstack(camera_augmented)
        joints_augmented = np.vstack(joints_augmented)
        targets_augmented = np.array(targets_augmented)  # Ensure correct shape

        print(f"Augmented Data Shapes: camera={camera_augmented.shape}, joints={joints_augmented.shape}, "
              f"waypoints={waypoints_augmented.shape}, targets={targets_augmented.shape}, "
              f"RGB images={rgb_images_augmented.shape}, Depth images={depth_images_augmented.shape}")

        return camera_augmented, joints_augmented, waypoints_augmented, targets_augmented, rgb_images_augmented, depth_images_augmented

    def vertical_flip_transform(self, image):
        """Apply vertical flip to an image."""
        return torch.flip(image, [1])


if __name__ == '__main__':
    dataset_dir = 'train_dataset'
    processor = DatasetProcessor(dataset_dir, 'train_dataset')
