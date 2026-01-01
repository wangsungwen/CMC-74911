.\yolo12_env\Scripts\activate.ps1
yolo train `
  model="yolo12n.pt" `
  data="datasets/wildlife/wildlife.yaml" `
  imgsz=640 epochs=100 batch=8 workers=8 `
  device=0 project="runs/train" name="dawn_supervised_v1" `
  degrees=0.0 `
  translate=0.08 `
  scale=0.3 `
  shear=0.0 `
  perspective=0.0 `
  mosaic=1.0 `
  copy_paste=0.35 `
  mixup=0.1 `
  hsv_h=0.015 `
  hsv_s=0.6 `
  hsv_v=0.45 `
  optimizer="AdamW" `
  lr0=0.001 `
  lrf=0.01 `
  momentum=0.9 `
  weight_decay=0.05 `
  warmup_epochs=3
