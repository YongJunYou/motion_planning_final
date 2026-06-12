"""Quick verify of the latest keyframe_reference.npz: tilt direction + box height near the window."""
import numpy as np

d = np.load("results/keyframe_reference.npz")
box, base, t = d["box"], d["base"], d["times"]
SPAWN = 1.5
kmax = int(np.argmax(box[:, 2]))
print(f"box peak z: home {box[kmax,2]:.2f} WORLD {box[kmax,2]+SPAWN:.2f} at x={box[kmax,0]:.2f} t={t[kmax]:.1f}s")
print(f"box z range world: {box[:,2].min()+SPAWN:.2f}..{box[:,2].max()+SPAWN:.2f}")
print(f"box x range world: {box[:,0].min():.2f}..{box[:,0].max():.2f}")
# base attitude (rotation vector) at the box-peak knot
if "grite_ref" in d:
    pass
# crossing of window plane x=-1
xs = box[:, 0]
for k in range(len(xs) - 1):
    if (xs[k] + 1.0) * (xs[k + 1] + 1.0) <= 0 and xs[k] != xs[k + 1]:
        f = (-1.0 - xs[k]) / (xs[k + 1] - xs[k])
        zc = box[k, 2] + f * (box[k + 1, 2] - box[k, 2]) + SPAWN
        yc = box[k, 1] + f * (box[k + 1, 1] - box[k, 1])
        print(f"box crosses window plane x=-1 at WORLD z={zc:.2f}, y={yc:.2f} (t~{t[k]:.1f}s)")
print("window opening: world z 2.06(bottom/slant)..3.07(top), |y|<0.865")
