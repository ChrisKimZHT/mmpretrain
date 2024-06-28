_base_ = [
    '../configs/_base_/default_runtime.py',
    './wuhan_scenery.py',
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
        num_classes=120,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0),
    ])
)

train_dataloader = dict(batch_size=32)

val_dataloader = dict(batch_size=32)

val_evaluator = dict(type='Accuracy', topk=(1, 3))

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='Adam',
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        amsgrad=False
    ),
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=20,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=180,
        by_epoch=True,
        begin=20,
        end=200,
    )
]

val_evaluator = dict(type='Accuracy', topk=(1, 3))

train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)

val_cfg = dict()

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1,
                    max_keep_ckpts=10, save_best="auto"),
)

visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ],
)
