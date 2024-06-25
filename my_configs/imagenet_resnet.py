_base_ = [
    '../configs/_base_/default_runtime.py',
    '../configs/_base_/schedules/imagenet_bs4096_AdamW.py',
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=100,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))

pipeline = [dict(type='LoadImageFromFile'), dict(
    type='Resize', scale=(224, 224)), dict(type='PackInputs')]

data_preprocessor = dict(
    num_classes=100,
    mean=[123.7284, 117.4751, 105.1077],
    std=[57.4626, 56.4878, 56.9013],
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
