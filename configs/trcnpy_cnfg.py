custom_imports = dict(
    imports=['datasets.tree_canopy'],
    allow_failed_imports=False
)

_base_ = [
    'C:/Users/harsh/Desktop/Keinisha/mmsegmentation/configs/_base_/models/fcn_unet_s5-d16.py',
    'C:/Users/harsh/Desktop/Keinisha/mmsegmentation/configs/_base_/default_runtime.py',
    'C:/Users/harsh/Desktop/Keinisha/mmsegmentation/configs/_base_/schedules/schedule_40k.py'
]

# ---------- Dataset ----------
dataset_type = 'TCD_Dataset'
data_root = 'data/training_data_object_detection/'

classes = ('individual', 'group_of_trees')
palette = [[255, 0, 0], [0, 0, 255]]

# ---------- Pipelines ----------
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53],
         std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='Pad', size_divisor=16, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize',
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375],
                 to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='Resize',
#         img_scale=(1024, 1024),
#         keep_ratio=False
#     ),
#     dict(
#         type='Normalize',
#         mean=[123.675, 116.28, 103.53],
#         std=[58.395, 57.12, 57.375],
#         to_rgb=True
#     ),
#     # dict(type='ImageToTensor', keys=['img']),
#     dict(
#         type='Collect',
#         keys=['img'],
#         meta_keys=['filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor']
#     )
# ]

# ---------- Dataloader ----------
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='TCD_Dataset',
        data_root=data_root,
        img_dir='images/train',
        ann_dir='mmseg_masks/train',
        pipeline=train_pipeline
    ),
    val=dict(
        type='TCD_Dataset',
        data_root=data_root,
        img_dir='images/val',
        ann_dir='mmseg_masks/val',
        pipeline=test_pipeline,
        test_mode=True
    ),
    test=dict(
        type='TCD_Dataset',
        data_root=data_root,
        img_dir='images/val',
        ann_dir='mmseg_masks/val',
        pipeline=test_pipeline,
        test_mode=True
    ),
)

# ---------- Model ----------
model = dict(
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# ---------- FP16 ----------
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
fp16 = dict()

# ---------- Runner (MMSeg 0.x) ----------
runner = dict(type='IterBasedRunner', max_iters=40000)

# ---------- Hooks ----------
checkpoint_config = dict(by_epoch=False, interval=2000)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])

# ---------- Evaluation ----------
evaluation = dict(interval=10000, metric='mIoU', pre_eval=True)