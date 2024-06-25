_base_ = [
    '../configs/_base_/default_runtime.py',
    '../configs/_base_/schedules/imagenet_bs4096_AdamW.py',
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        arch='b',
        img_size=224,
        patch_size=32,
        drop_rate=0.1,
        init_cfg=[
            dict(
                type='Kaiming',
                layer='Conv2d',
                mode='fan_in',
                nonlinearity='linear')
        ]),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=100,
        in_channels=768,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))


pipeline = [dict(type='LoadImageFromFile'), dict(
    type='Resize', scale=(224, 224)), dict(type='PackInputs')]

data_preprocessor = dict(
    num_classes=100,
    mean=[123.7504, 117.504, 105.1392],
    std=[58.9824, 58.0352, 58.4192],
    to_rgb=True,
)

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    dataset=dict(
        type='CustomDataset',
        data_root='/home/chriskim/ILSVRC2012',
        data_prefix='train',
        pipeline=pipeline
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
)

val_dataloader = dict(
    batch_size=32,
    num_workers=8,
    dataset=dict(
        type='CustomDataset',
        data_root='/home/chriskim/ILSVRC2012',
        data_prefix='val',
        pipeline=pipeline
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
)

val_evaluator = dict(type='Accuracy', topk=(1, 5))

train_cfg = dict(max_epochs=100)

test_cfg = None
