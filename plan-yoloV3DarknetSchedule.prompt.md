## Plan: YOLOv3 Darknet-Style Training Schedule

You currently train via Ultralytics (no custom PyTorch loop/scheduler in-repo), so the cleanest path is to add a new script that either (A) stays with Ultralytics but forces a step-based LR schedule, or (B) introduces a fully custom PyTorch loop where “20,000 steps” is first-class. Your epoch mapping is purely math: epochs $= \frac{20000 \cdot 64}{N}$ where $N$ is the number of training images.

### Steps
1. Decide implementation route: Ultralytics wrapper vs fully custom PyTorch YOLOv3 loop.
2. Compute epoch equivalent from dataset size: epochs $= 20000 \times 64 / N$ (use `N = len(train_dataset)`).
3. Implement global-step-driven loop: iterate dataloader indefinitely until `global_step == 20000`.
4. Add SGD optimizer with `lr=0.001` and momentum/weight decay matching the paper (if specified).
5. Implement custom scheduler: `lr = lr0 * decay_factor ** (global_step // 4000)` or continuous exponential form.
6. Wire logging/checkpointing so step-based schedule is reproducible (save `global_step`, optimizer, scheduler state).

### Further Considerations
1. What is the LR decay factor every 4,000 steps (e.g., `0.1`, `0.5`, or paper-specific)?
2. What is your training set size `N` (number of train images in `face_datasets/train/images`) so epochs can be computed numerically?
3. Do you need “true YOLOv3” (Darknet-style model + losses), or only the schedule replicated on your current YOLO (Ultralytics)?
