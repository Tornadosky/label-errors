from loss_function import detection_loss

import json
import os
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    train_data_dir = 'dataset/bdd100k-DatasetNinja/train/img'
    test_data_dir = 'dataset/bdd100k-DatasetNinja/test/img'
    val_data_dir = 'dataset/bdd100k-DatasetNinja/val/img'

    # load data
    train_dataset = CocoDetection(train_data_dir, annFile='input/train2.json', transform=transforms.ToTensor())
    test_dataset = CocoDetection(test_data_dir, annFile='input/test2.json', transform=transforms.ToTensor())
    val_dataset = CocoDetection(val_data_dir, annFile='input/val2.json', transform=transforms.ToTensor())

    # model
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
    model = FasterRCNN(backbone, num_classes=12, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)

    # def
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)

    # optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # same size
    height = 720
    width = 1280

    # training loop
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        for images, targets in train_loader:
            images = [image.to(device) for image in images]

            # images_with_info = [[image, height, width] for image in images]

            targets = [
                [
                    {
                        'id': target['id'],
                        'image_id': target['image_id'],
                        'boxes': target['bbox'],
                        'labels': target['category_id']
                    }
                    for target in targets_per_image
                ]
                for targets_per_image in targets
            ]

            boxes = []
            labels = []
            image_id = 0
            new_goat = []

            for i in range(len(targets)):
                boxes_inter = []
                labels_inter = []
                image_id_inter = 0
                for target in targets[i]:
                    zizi = (target['boxes'])
                    zozo = (target['labels'])
                    # print(zizi)
                    # torch.tensor(zizi)
                    # print(zizi)
                    boxes_inter.append(zizi)
                    labels_inter.append(zozo)
                    image_id_inter = target['image_id']

                # boxes.append(torch.tensor(boxes_inter))
                # labels.append(torch.tensor(labels_inter))
                # image_id = torch.tensor(image_id_inter)

                boxes.append((boxes_inter))
                labels.append((labels_inter))
                image_id = (image_id_inter)

                goat_inter = {'boxes': boxes,
                              'labels': labels,
                              'image_id': torch.tensor(image_id)}

                boxes = []
                labels = []
                image_id = 0

                new_goat.append(goat_inter)

            #for target in new_goat:
            #    for d in target:
            #        d['labels'] = d['labels'].squeeze()


            new_targets = [
                {
                    'boxes': torch.tensor(target['boxes']),
                    'labels': torch.tensor(target['labels']),
                    # 'image_id': target['image_id']
                }
                for target in new_goat
            ]

            for target in new_targets:
                target['boxes'] = target['boxes'].view(-1, 4)
                target['labels'] = target['labels'].view(-1)

            #print((images))
            #print((new_targets))

            images = [image.to(device) for image in images]
            new_targets = [{k: v.to(device) for k, v in t.items()} for t in new_targets]

            # int64 pls
            for target in new_targets:
                target['labels'] = target['labels'].long()

            # print(new_targets)
            outputs = model(images, new_targets)
            #print(outputs)

            # sans tenseur
            #predicted_classes = outputs['class_predictions']
            #predicted_boxes = outputs['box_predictions']

            #true_classes = [target['labels'] for target in targets]
            #true_boxes = [target['boxes'] for target in targets]

            #loss = detection_loss(predicted_classes, true_classes, predicted_boxes, true_boxes)
            # fonction de detection loss
            loss = outputs['loss_classifier'] + outputs['loss_box_reg']

            # avec tenseur
            # predicted_classes_tensor = torch.tensor(predicted_classes, dtype=torch.float32)
            # predicted_boxes_tensor = torch.tensor(predicted_boxes, dtype=torch.float32)

            # true_classes_tensor = [torch.tensor(target['labels'], dtype=torch.long) for target in targets]
            # true_boxes_tensor = [torch.tensor(target['boxes'], dtype=torch.float32) for target in targets]

            # loss = detection_loss(predicted_classes_tensor, true_classes_tensor, predicted_boxes_tensor, true_boxes_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_scheduler.step()
            print("Loss")

            # torch.cuda.empty_cache()

    # Évaluation du modèle sur les données de test
    print("finished")

    #model.eval()

    # Ssave
    torch.save(model.state_dict(), 'object_detection_model.pth')


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()
