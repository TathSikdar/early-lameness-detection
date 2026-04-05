from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Paths — update these to match your local setup
# ---------------------------------------------------------------------------
DATA_YAML       = '../../EarTagModel/Pose/cattleeyeview_pose_copy.yaml'

# FIX 1: Upgrade from nano → small.  The nano model only has ~3 M params which
# is too constrained for 24-keypoint cow pose.  The small model (~11 M params)
# gives significantly better keypoint accuracy at modest extra cost.
PRETRAINED_MODEL = '../../EarTagModel/Pose/yolo26s-pose.pt'
CONTINUE_TRAINING_MODEL = 'runs/pose/runs/pose/cow_pose_finetune/train5/weights/last.pt'

OUTPUT_DIR = 'runs/pose/cow_pose_finetune'

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
# FIX 2: Increase image resolution.  Original footage is 1920×1080 but the
# previous run used 640 px — too aggressive a downscale for subtle keypoints
# (paws, ears, etc.).  1280 preserves far more spatial detail.  Halve the
# batch size to keep GPU memory usage reasonable.
IMG_SIZE   = 1280
BATCH_SIZE = 8

# FIX 3: More epochs + cosine LR schedule.  Val/pose_loss was still 4.38 at
# epoch 100 (vs train 1.85) — the model had not converged and was already
# overfitting.  Cosine annealing gives a smoother final descent and often
# recovers an extra 1-2 % mAP at the end.
EPOCHS     = 10
PATIENCE   = 50   # stop early if val pose loss stops improving


def main():
    model = YOLO(PRETRAINED_MODEL)
    # model = YOLO(CONTINUE_TRAINING_MODEL)
    
    model.train(
        data        = DATA_YAML,
        epochs      = EPOCHS,
        patience    = PATIENCE,
        batch       = BATCH_SIZE,
        imgsz       = IMG_SIZE,
        project     = OUTPUT_DIR,
        degrees     = 30.0,       # ±15° rotation
        scale       = 0.5,        # random scale in [0.5, 1.5]
        translate   = 0.1,
        hsv_h       = 0.015,
        hsv_s       = 0.5,
        hsv_v       = 0.4,
        device      = 0
    )

# # FIX 4: Cosine LR decay — smoother convergence than linear.
#         cos_lr      = True,
#         lr0         = 0.01,
#         lrf         = 0.005,      # final LR = lr0 × lrf = 5e-5

#         # FIX 5: Dropout regularisation to combat overfitting.  Val/pose_loss
#         # was 2.4× the training loss — a clear sign the model was memorising
#         # rather than generalising.
#         dropout     = 0.1,

#         # FIX 6: Augmentation tuning for overhead cattle footage.
#         # - Gentle rotation: cows can face any direction in a top-down view.
#         # - Scale jitter: cows appear at different distances from the camera.
#         # - No fliplr/flipud beyond YOLO defaults because flip_idx is already
#         #   defined in the yaml and will be applied automatically.
#         degrees     = 15.0,       # ±15° rotation
#         scale       = 0.5,        # random scale in [0.5, 1.5]
#         translate   = 0.1,
#         hsv_h       = 0.015,
#         hsv_s       = 0.5,
#         hsv_v       = 0.4,
#         # Keep mosaic on (default=1.0) but close it later in training so the
#         # model also sees clean single-instance frames.
#         close_mosaic = 20,

#         # FIX 7: Multi-scale training adds scale variety without extra data —
#         # helps generalise to cows at different heights in the frame.
#         multi_scale = True,

#         # FIX 8: Weight decay — additional regularisation on top of dropout.
#         weight_decay = 0.001,

#         # Misc
#         workers     = 8,
#         seed        = 42,
#         plots       = True,

if __name__ == '__main__':
    main()