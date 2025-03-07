import os
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from PIL import Image


class TestDatasetProcessor:
    def __init__(self, dataset_dir, base_waypoint_path):
        self.dataset_dir = dataset_dir
        self.base_waypoint_path = os.path.abspath(
            os.path.join(r'Path/to/project', base_waypoint_path)
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

    def load_waypoints_data(self, waypoint_path):

        try:
            waypoint_data = pd.read_csv(waypoint_path)
            return waypoint_data.values
        except Exception as e:
            print(f"can not load file {waypoint_path}: {e}")
            return np.zeros((1, 301))

    def load_and_process_image(self, image_path, is_depth=False):

        try:
            image = Image.open(image_path).convert('RGB' if not is_depth else 'L')  # Convert to grayscale if depth
            image = self.image_transform(image) if not is_depth else self.depth_transform(image)
            return image
        except Exception as e:
            print(f"can not load image {image_path}: {e}")
            return torch.zeros(3, 240, 320) if not is_depth else torch.zeros(1, 240, 320)

    def process_dataset(self):
        camera_values_list = []
        joint_states_list = []
        waypoints_data_list = []
        targets_list = []
        rgb_images_list = []
        depth_images_list = []

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
                # Extract features
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
                                   'panda_joint7']].values

                filtered_labels = df.loc[valid_rows, 'Category'].map(self.label_mapping).values.astype(np.int64)

                filtered_camera = flattened_camera_values[valid_rows]
                filtered_joints = joint_states[valid_rows]

                waypoints_data = []
                for wp_path in df.loc[valid_rows, 'Waypoints']:
                    full_path = os.path.join(subfolder_path, wp_path)
                    full_path = full_path.replace('/test_dataset', self.base_waypoint_path)
                    wp = self.load_waypoints_data(full_path)
                    waypoints_data.append(wp.flatten())

                for rgb_path, depth_path in zip(df.loc[valid_rows, 'RGB_image'], df.loc[valid_rows, 'Depth_image']):
                    full_rgb_path = os.path.join(subfolder_path, rgb_path)
                    full_depth_path = os.path.join(subfolder_path, depth_path)

                    rgb_images_list.append(self.load_and_process_image(full_rgb_path, is_depth=False))
                    depth_images_list.append(self.load_and_process_image(full_depth_path, is_depth=True))

                camera_values_list.append(filtered_camera)
                joint_states_list.append(filtered_joints)
                waypoints_data_list.append(np.array(waypoints_data))
                targets_list.append(filtered_labels)

            except Exception as e:
                print(f"deal {subfolder} fails: {e}")

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

        return camera_values, joint_states, waypoints_data, targets, rgb_images, depth_images

    def save_to_csv(self, camera, joints, waypoints, targets, rgb_paths, depth_paths, waypoint_paths,
                    output_path='test_data.csv'):
        """Save processed data along with image and waypoint file paths to a CSV file."""
        df = pd.DataFrame()

        # Add features
        for i in range(75):
            df[f'feature_{i}'] = camera[:, i]
        for i in range(7):
            df[f'joint_{i}'] = joints[:, i]
        for i in range(waypoints.shape[1]):
            df[f'waypoint_{i}'] = waypoints[:, i]
        df['label'] = targets

        # Add file paths
        df['rgb_image_path'] = rgb_paths
        df['depth_image_path'] = depth_paths
        df['waypoint_file_path'] = waypoint_paths

        df.to_csv(output_path, index=False)
        print(f"save data for check: {output_path}")

    def save_to_tensors(self, camera, joints, waypoints, targets, rgb_images, depth_images,
                        output_dir='processed_data'):
        """Save data as PyTorch tensors."""
        os.makedirs(output_dir, exist_ok=True)

        torch.save(camera, os.path.join(output_dir, 'camera.pt'))
        torch.save(joints, os.path.join(output_dir, 'joints.pt'))
        torch.save(waypoints, os.path.join(output_dir, 'waypoints.pt'))
        torch.save(targets, os.path.join(output_dir, 'targets.pt'))
        torch.save(rgb_images, os.path.join(output_dir, 'rgb_images.pt'))
        torch.save(depth_images, os.path.join(output_dir, 'depth_images.pt'))

        print(f"Save data to {output_dir}")


if __name__ == '__main__':
    dataset_dir = 'test_dataset'
    processor = TestDatasetProcessor(dataset_dir, 'test_dataset')
