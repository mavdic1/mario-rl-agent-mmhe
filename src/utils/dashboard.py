import cv2
import numpy as np
import torch

def show_mario_dashboard(obs, infos, rewards, actions, model, num_timesteps, version="v2"):
    """
    Renders the Master Research Dashboard in a single window.
    """
    if obs is None or len(infos) == 0:
        return

    # --- DATA EXTRACTION (Env 0) ---
    info = infos[0]
    curr_x = info.get("max_x", 0)
    stuck = info.get("stuck_timer", 0)
    rew = rewards[0] if rewards is not None else 0
    obs_tensor = torch.as_tensor(obs).to(model.device).float()

    # --- BRAIN LOGIC (VALUE & NEURAL) ---
    with torch.no_grad():
        # Value Estimate (Optimism)
        value_est = model.policy.predict_values(obs_tensor)[0].item()
        
        # Neural Activations (Heatmap)
        # Assumes NatureCNN structure from Stable Baselines 3
        cnn_layer1 = model.policy.features_extractor.cnn[0]
        activations = cnn_layer1(obs_tensor[0:1])
        heatmap = torch.mean(activations[0], dim=0).cpu().numpy()
        
        # Neural Weights (Filters) - first 8
        weights = cnn_layer1.weight[0:8, 0, :, :].cpu().numpy()

    # --- VISUAL PROCESSING ---
    # Main Game View (Motion Ghosting - mean of stacked frames)
    ghost_frame = np.mean(obs[0], axis=0).astype(np.uint8)
    main_view = cv2.resize(ghost_frame, (350, 350), interpolation=cv2.INTER_NEAREST)
    main_view = cv2.cvtColor(main_view, cv2.COLOR_GRAY2BGR)

    # Heatmap Processing
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_img = cv2.resize(heatmap, (180, 180))
    heatmap_img = cv2.applyColorMap((heatmap_img * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Filters Processing (Grid of 8)
    weight_grid = []
    for w in weights:
        w_norm = (w - w.min()) / (w.max() - w.min() + 1e-8)
        w_img = cv2.resize(w_norm, (35, 35), interpolation=cv2.INTER_NEAREST)
        weight_grid.append((w_img * 255).astype(np.uint8))
    row1 = np.hstack(weight_grid[0:4])
    row2 = np.hstack(weight_grid[4:8])
    filters_img = cv2.cvtColor(np.vstack([row1, row2]), cv2.COLOR_GRAY2BGR)

    # --- BUILD CONSOLIDATED CANVAS ---
    # Total Width: 350 (Game) + 200 (Sidebar) = 550
    # Total Height: 60 (Top) + 350 (Mid) + 60 (Bottom) = 470
    canvas = np.zeros((470, 550, 3), dtype=np.uint8)

    # 1. Place Main View
    canvas[60:410, 0:350] = main_view

    # 2. Place Sidebar Visuals
    cv2.putText(canvas, "NEURAL ACTIVATIONS", (365, 80), 0, 0.4, (255, 255, 255), 1)
    canvas[90:270, 360:540] = heatmap_img
    cv2.putText(canvas, "LEARNED FILTERS", (365, 300), 0, 0.4, (255, 255, 255), 1)
    canvas[315:385, 380:520] = filters_img

    # --- HUD & TEXT OVERLAYS ---
    # Header Bar
    cv2.rectangle(canvas, (0, 0), (550, 60), (30, 30, 30), -1)
    cv2.putText(canvas, f"STEP: {num_timesteps // 1000}K", (10, 35), 0, 0.5, (200, 200, 200), 1)
    cv2.putText(canvas, f"X: {int(curr_x)}", (140, 40), 0, 0.8, (0, 255, 0), 2)
    v_color = (0, 255, 0) if value_est > 0 else (0, 0, 255)
    cv2.putText(canvas, f"VAL: {value_est:.1f}", (380, 38), 0, 0.6, v_color, 2)

    # Bottom Dashboard Bar
    cv2.rectangle(canvas, (0, 410), (550, 470), (20, 20, 20), -1)
    
    # Action Mapping
    action_names = ["IDLE", "RIGHT", "R+JUMP", "R+RUN", "R+R+J", "JUMP", "LEFT"]
    act_idx = actions[0] if actions is not None else 0
    act_str = action_names[act_idx] if act_idx < len(action_names) else str(act_idx)
    cv2.putText(canvas, f"INPUT: {act_str}", (10, 435), 0, 0.5, (0, 255, 255), 2)
    
    # Stuck Timer
    s_color = (0, 0, 255) if stuck > 150 else (255, 255, 255)
    cv2.putText(canvas, f"STUCK: {stuck}/250", (10, 458), 0, 0.4, s_color, 1)

    # Reward
    cv2.putText(canvas, f"REW: {rew:.1f}", (200, 435), 0, 0.5, (255, 255, 255), 1)

    # Progress Bar (Yellow)
    progress = min(curr_x / 6400, 1.0) 
    cv2.rectangle(canvas, (200, 445), (530, 455), (50, 50, 50), -1)
    cv2.rectangle(canvas, (200, 445), (200 + int(330 * progress), 455), (0, 255, 255), -1)
    
    # Add Level text to Dashboard
    world = info.get("world", 1)
    lvl = info.get("level", 1)
    cv2.putText(canvas, f"LVL: {world}-{lvl}", (250, 35), 0, 0.5, (255, 255, 255), 1)

    # --- DISPLAY ---
    cv2.imshow(f"Mario AI Dashboard - {version.upper()}", canvas)
    cv2.waitKey(1)