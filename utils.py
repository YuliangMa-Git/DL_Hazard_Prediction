import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def loss_fn(pred, y):
    outputs = pred.squeeze(1)
    loss = torch.nn.functional.binary_cross_entropy(outputs, y)
    return loss


def model_evaluation(data_loader, model, device):
    gt_list = []
    preds_list = []
    test_loss = 0.0
    model.eval()

    with torch.no_grad():
        for (camera_values, joint_states, waypoints_data, label, rgb_images, depth_images) in data_loader:
            B = camera_values.shape[0]
            camera_values = camera_values.unsqueeze(1)
            camera_values = camera_values.reshape(B, 25, 3).to(device)

            waypoints_data = waypoints_data.unsqueeze(1)
            waypoints_data = waypoints_data.reshape(B, 64, 7).to(device)

            joint_states = joint_states.to(device)
            rgb_images = rgb_images.to(device)
            depth_images = depth_images.to(device)
            label = label.to(device)
            pred = model(camera_values, joint_states, waypoints_data)
            # pred = model(rgb_images, depth_images, joint_states, waypoints_data)

            gt_list.extend(list(label.cpu().flatten()))
            preds_list.extend(list(pred.cpu().numpy().flatten()))

            loss = loss_fn(pred, label)
            test_loss += loss.item()
        test_loss /= len(data_loader)
    return test_loss


def get_F1_measure(data_loader, model, device, threshold):
    gt_list = []
    preds_list = []
    model.eval()
    inference_times = []
    with torch.no_grad():
        for (camera_values, joint_states, waypoints_data, label, rgb_images, depth_image) in data_loader:
            B = camera_values.shape[0]
            camera_values = camera_values.unsqueeze(1)
            camera_values = camera_values.reshape(B, 25, 3).to(device)

            waypoints_data = waypoints_data.unsqueeze(1)
            waypoints_data = waypoints_data.reshape(B, 64, 7).to(device)

            joint_states = joint_states.to(device)

            rgb_images = rgb_images.to(device)
            depth_images = depth_image.to(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            pred = model(camera_values, joint_states, waypoints_data)
            # pred = model(rgb_images, depth_images, joint_states, waypoints_data)
            end_event.record()

            torch.cuda.synchronize()  # Ensure all GPU work is done

            inference_time = start_event.elapsed_time(end_event)  # Time in milliseconds
            inference_times.append(inference_time / B)  # Store per-sample time

            print(f"Inference time for batch: {inference_time:.3f} ms, per sample: {inference_time / B:.3f} ms")
            pred = pred.squeeze(1)

            pred_label = pred > threshold
            gt_list.extend(list(label.flatten()))
            preds_list.extend(list(pred_label.cpu().numpy().flatten()))

    avg_inference_time = sum(inference_times) / len(inference_times)
    print(f"\nAverage inference time per sample: {avg_inference_time:.3f} ms")

    tn, fp, fn, tp = confusion_matrix(gt_list, preds_list).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = (2 * recall * precision) / (recall + precision)

    print("precision is: ", precision)
    print("recall is: ", recall)
    print("f1 score is: ", f1_score)

    cm = confusion_matrix(gt_list, preds_list)
    class_labels = ['Safety', 'Hazard']

    # Create a confusion matrix plot
    plt.figure(figsize=(24, 18))
    csfont = {'fontname': 'Times New Roman'}
    sns.set(font_scale=12, font="Times New Roman")
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g',
                xticklabels=class_labels, yticklabels=class_labels, cbar=False, vmin=0, vmax=150)
    plt.savefig('confusion_matrix')  # Save the plot as a PNG file
    # plt.show()
    return
