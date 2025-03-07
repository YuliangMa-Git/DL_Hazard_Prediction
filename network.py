import torch
import torch.nn as nn
import torch.nn.functional as F


class CnnModel(nn.Module):
    def __init__(self, input_length=64, num_classes=2):
        super(CnnModel, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=7, out_channels=16, kernel_size=7, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(32)

        self.flatten_dim = self._get_flatten_dim(input_length)
        self.fc1 = nn.Linear(self.flatten_dim, 256)

    def _get_flatten_dim(self, input_length):
        x = torch.zeros(1, 7, input_length)
        x = F.max_pool1d(F.relu(self.bn1(self.conv1(x))), kernel_size=2)
        x = F.max_pool1d(F.relu(self.bn2(self.conv2(x))), kernel_size=2)
        return x.numel()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.bn1(self.conv1(x))
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2)
        x = self.bn2(self.conv2(x))
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2)

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return x


class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, image_feature):
        image_feature = self.bn1(self.conv1(image_feature))
        image_feature = self.pool1(nn.functional.relu(image_feature))
        image_feature = self.bn2(self.conv2(image_feature))
        image_feature = self.pool2(nn.functional.relu(image_feature))
        image_feature = self.bn3(self.conv3(image_feature))
        image_feature = self.pool3(nn.functional.relu(image_feature))

        return image_feature


class JointStatesModel(nn.Module):
    def __init__(self):
        super(JointStatesModel, self).__init__()
        self.fc_L_1 = nn.Linear(7, 128)
        self.fc_A_1 = nn.ReLU()
        self.fc_L_2 = nn.Linear(128, 256)
        self.fc_A_2 = nn.ReLU()

    def forward(self, joint_feature):
        joint_feature = self.fc_A_1(self.fc_L_1(joint_feature))
        joint_feature = self.fc_A_2(self.fc_L_2(joint_feature))
        return joint_feature


class CameraModel(nn.Module):
    def __init__(self):
        super(CameraModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, stride=1,
                               padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1,
                               padding=1)
        self.relu2 = nn.ReLU()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(32, 256)

    def forward(self, camera_feature):
        camera_feature = camera_feature.permute(0, 2, 1)

        camera_feature = self.relu1(self.conv1(camera_feature))
        camera_feature = self.relu2(self.conv2(camera_feature))

        camera_feature = self.global_avg_pool(camera_feature)

        camera_feature = camera_feature.squeeze(-1)
        camera_feature = self.fc1(camera_feature)
        return camera_feature


class HazardPredictionImageNN(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        self.image_model = ImageModel()
        self.waypoints_model = CnnModel()
        self.joint_model = JointStatesModel()

        self.img_L1 = nn.Linear(640, 128)
        self.img_A1 = nn.ReLU()

        self.waypoint_L1 = nn.Linear(256, 64)
        self.waypoint_A1 = nn.ReLU()

        self.joint_L1 = nn.Linear(256, 64)
        self.joint_A1 = nn.ReLU()

        self.fc_L_1 = nn.Linear(256, 128)
        self.fc_A_1 = nn.ReLU()

        self.dropout = nn.Dropout(p=0.5)

        self.fc_L_2 = nn.Linear(128, 32)
        self.fc_A_2 = nn.ReLU()

        self.fc_L_3 = nn.Linear(32, num_classes)
        self.fc_A_3 = nn.Sigmoid()

    def forward(self, img, depth, joint, waypoint):
        early_fused_features = torch.cat((img, depth), dim=1)
        image_features = self.image_model(early_fused_features)
        joint_features = self.joint_model(joint)
        waypoint_features = self.waypoints_model(waypoint)

        image_features = torch.flatten(image_features, 1)
        joint_features = self.joint_A1(self.joint_L1(joint_features))
        waypoint_features = self.waypoint_A1(self.waypoint_L1(waypoint_features))

        image_features = self.img_A1(self.img_L1(image_features))

        cat_features = torch.cat((image_features, joint_features, waypoint_features), dim=1)
        cat_features = self.fc_A_1(self.fc_L_1(cat_features))
        cat_features = self.dropout(cat_features)
        cat_features = self.fc_A_2(self.fc_L_2(cat_features))
        cat_features = self.dropout(cat_features)
        pred = self.fc_A_3(self.fc_L_3(cat_features))
        return pred


class HazardPredictionNN(nn.Module):

    def __init__(self, num_classes=1):
        super().__init__()
        self.joint_model = JointStatesModel()
        self.waypoints_model = CnnModel()
        self.camera_model = CameraModel()

        self.joint_L1 = nn.Linear(256, 64)
        self.joint_A1 = nn.ReLU()
        self.camera_L1 = nn.Linear(256, 128)
        self.camera_A1 = nn.ReLU()
        self.waypoint_L1 = nn.Linear(256, 64)
        self.waypoint_A1 = nn.ReLU()
        self.fc_L_1 = nn.Linear(256, 128)
        self.fc_A_1 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc_L_2 = nn.Linear(128, 32)
        self.fc_A_2 = nn.ReLU()
        self.fc_L_3 = nn.Linear(32, num_classes)
        self.fc_A_3 = nn.Sigmoid()

    def forward(self, camera, joint, waypoint):
        camera_features = self.camera_model(camera)
        joint_features = self.joint_model(joint)

        waypoint_features = self.waypoints_model(waypoint)

        camera_features = self.camera_A1(self.camera_L1(camera_features))
        joint_features = self.joint_A1(self.joint_L1(joint_features))
        waypoint_features = self.waypoint_A1(self.waypoint_L1(waypoint_features))

        cat_features = torch.cat((camera_features, joint_features, waypoint_features), dim=1)
        cat_features = self.fc_A_1(self.fc_L_1(cat_features))
        cat_features = self.dropout(cat_features)
        cat_features = self.fc_A_2(self.fc_L_2(cat_features))
        cat_features = self.dropout(cat_features)
        pred = self.fc_A_3(self.fc_L_3(cat_features))

        return pred
