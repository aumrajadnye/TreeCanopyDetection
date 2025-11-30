custom_imports = dict(
    imports=['datasets.tree_canopy', 'datasets.debug_pipeline'],
    allow_failed_imports=False
)

log_processor=dict(by_epoch=False, window_size=1)

# ---------- Base Configs ----------
_base_ = [
    'C:/Users/keini/OneDrive/Desktop/Code/mmsegmentation/configs/_base_/models/fcn_unet_s5-d16.py',
    'C:/Users/keini/OneDrive/Desktop/Code/mmsegmentation/configs/_base_/default_runtime.py', 
    'C:/Users/keini/OneDrive/Desktop/Code/mmsegmentation/configs/_base_/schedules/schedule_40k.py'
]

# ---------- Dataset ----------
dataset_type = 'TCD_Dataset'
data_root = 'data/training_data_object_detection/'

classes = ('individual', 'group_of_trees')
palette = [
    [255,0,0],      # individual ─ red
    [0,0,255],      # group_of_trees ─ blue
]

# crop_size = (1024, 1024)

# ---------- Pipelines ----------
train_pipeline = [
    dict(type='DebugPrint', msg='Start Train Pipeline'),
    dict(type='LoadImageFromFile'),
    dict(type='DebugPrint', msg='After LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='DebugPrint', msg='After LoadAnnotations'),
    dict(type='RandomResize', scale=(512,512), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='DebugPrint', msg='After RandomResize'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='DebugPrint', msg='After RandomFlip'),
    dict(type='PackSegInputs'),
    dict(type='DebugPrint', msg='After PackSegInputs'),
]

test_pipeline = [
    dict(type='DebugPrint', msg='Start Test Pipeline'),
    dict(type='LoadImageFromFile'),
    dict(type='DebugPrint', msg='After LoadImageFromFile'),
    dict(type='Resize', scale=(512,512), keep_ratio=True),
    dict(type='DebugPrint', msg='After Resize'),
    dict(type='LoadAnnotations'),
    dict(type='DebugPrint', msg='After LoadAnnotations'),
    dict(type='PackSegInputs'),
    dict(type='DebugPrint', msg='After PackSegInputs'),
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]

# ---------- Dataloader ----------
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    # sampler=dict(type='DefaultSampler', shuffle=True),
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/train', seg_map_path='mmseg_masks/train'),
        pipeline=train_pipeline,
        img_suffix='.png',
        seg_map_suffix='.png'
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    # sampler=dict(type='InfiniteSampler', shuffle=True),
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/val', seg_map_path='mmseg_masks/val'),
        pipeline=test_pipeline,
        img_suffix='.png',
        seg_map_suffix='.png'
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=True,
    # sampler=dict(type='InfiniteSampler', shuffle=True),
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/train'),
        pipeline=test_pipeline,
        img_suffix='.png',
        seg_map_suffix='.png'
    )
)

print("DEBUG: Creating train dataset...")
print(train_dataloader['dataset'])
print("DEBUG: Creating val dataset...")
print(val_dataloader['dataset'])
print("DEBUG: Creating test dataset...")
print(test_dataloader['dataset'])

# ----------- Evaluation Loops -----------
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator


# ---------- U-Net Model ----------
model = dict(
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2)
)

# ---------- Training ----------
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=40000,
    val_interval=2000,
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optim_wrapper = dict(
#     _delete_=True,
#     optimizer=dict(type='Adam', lr=0.0001),
#     clip_grad=dict(max_norm=35, norm_type=2)
# )

# ---------- Checkpoint Saving ----------
dist_cfg = dict(backend='gloo')

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,         # save every 2000 iterations
        by_epoch=False,        # since using iter-based loop
        save_best='mIoU'       # best checkpoint based on validation mIoU
    )
)

default_hooks.update({
    'logger': dict(type='LoggerHook', interval=1),
})

vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='SegLocalVisualizer', 
    vis_backends=vis_backends, 
    name='visualizer'
)